"""The Multi-Target Tracking engine: one continuous predict/gate/associate/
update/lifecycle loop per radar sweep.

    for each sweep(t, pings):
        1. predict every track forward to t
        2. gate pings against predicted tracks (spatial index + Mahalanobis)
        3. associate (sub-graph fractured Hungarian on the innovation-cov cost)
        4. update matched tracks; coast the missed ones
        5. spawn tentative tracks from unassigned pings
        6. promote / prune per the lifecycle state machine

Detections arrive already in the metric ENU frame. The engine is frame-
agnostic; geo.LocalFrame handles WGS84<->ENU at the I/O edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gating, kalman
from .association import associate
from .track import LifecycleParams, TrackBank


@dataclass
class TrackerConfig:
    process_noise_q: float = 5.0          # accel PSD (m^2/s^3)
    meas_noise_std: float = 30.0          # per-axis measurement std (m)
    gate_threshold: float = gating.CHI2_GATE_99
    use_hash_grid: bool = False           # KDTree by default; True -> HashGrid
    lifecycle: LifecycleParams = None

    def __post_init__(self):
        if self.lifecycle is None:
            self.lifecycle = LifecycleParams()


class Tracker:
    @classmethod
    def for_scenario(cls, scn) -> "Tracker":
        """Build a Tracker tuned to a scenario's noise + recommended process q."""
        return cls(TrackerConfig(
            meas_noise_std=scn.meta.get("meas_std", 30.0),
            process_noise_q=scn.meta.get("q_hint", 5.0),
        ))

    def __init__(self, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()
        self.R = np.eye(3) * (self.cfg.meas_noise_std ** 2)
        self.bank = TrackBank(self.cfg.lifecycle)
        self._last_t: float | None = None
        self.frame_index = 0

    def step(self, t: float, pings: np.ndarray) -> dict:
        """Process one sweep. `pings` is (M, 3) ENU detections. Returns a report."""
        pings = np.asarray(pings, dtype=float).reshape(-1, 3)
        dt = 1.0 if self._last_t is None else max(t - self._last_t, 1e-3)
        self._last_t = t

        # 1. predict
        self.bank.predict(dt, self.cfg.process_noise_q)

        # 2. gate
        index_cls = gating.HashGrid if self.cfg.use_hash_grid else gating.KDTreeIndex
        gated = gating.gate(self.bank.x, self.bank.P, pings, self.R,
                            self.cfg.gate_threshold, index_cls=index_cls)

        # innovation log-det for the NLL cost
        S = kalman.innovation_covariance(self.bank.P, self.R)
        logdetS = (np.linalg.slogdet(S)[1] if len(S) else np.empty(0))

        # 3. associate
        matches, unmatched_tracks, _ = associate(
            gated, self.cfg.gate_threshold, use_loglik=True, logdetS=logdetS)

        # 4. update matched, coast missed
        if matches.shape[0]:
            track_idx = matches[:, 0]
            ping_idx = matches[:, 1]
            self.bank.apply_updates(track_idx, pings[ping_idx], self.R, t)
            assigned_pings = set(int(m) for m in ping_idx)
        else:
            assigned_pings = set()

        # tracks that exist but got no assignment -> coast
        all_track_idx = np.arange(self.bank.count)
        matched_set = set(int(k) for k in (matches[:, 0] if matches.shape[0] else []))
        missed = np.asarray([k for k in all_track_idx if k not in matched_set],
                            dtype=int)
        self.bank.mark_missed(missed, t)

        # 5. spawn from unassigned pings
        unassigned = np.asarray(
            [m for m in range(pings.shape[0]) if m not in assigned_pings], dtype=int)
        if unassigned.size:
            self.bank.spawn(pings[unassigned], t)

        # 6. lifecycle
        self.bank.promote()
        deleted_ids = self.bank.prune()

        self.frame_index += 1
        return {
            "t": t,
            "frame": self.frame_index,
            "n_pings": int(pings.shape[0]),
            "n_matched": int(matches.shape[0]),
            "n_spawned": int(unassigned.size),
            "n_deleted": int(deleted_ids.size),
            "n_tracks": self.bank.count,
            "n_confirmed": int(self.bank.confirmed_mask().sum()),
            "tracks": self.bank.snapshot(),
        }

    def run(self, sweeps):
        """Run over an iterable of (t, pings) sweeps, yielding per-frame reports."""
        for t, pings in sweeps:
            yield self.step(t, pings)
