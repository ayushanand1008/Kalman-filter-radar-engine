"""Track lifecycle state machine and the structure-of-arrays TrackBank.

Tracks are kept as parallel numpy arrays (structure-of-arrays) so the Kalman
predict/update steps operate on the whole bank at once. Per-track scalar
metadata (id, status, hit/miss counters) live alongside the state arrays.

Lifecycle:
    TENTATIVE  -- freshly spawned from an unassigned ping
    CONFIRMED  -- promoted after M hits within the confirmation window
    (coasting) -- a *confirmed* track that missed the current sweep; kept alive
                  on kinematic prediction. Represented by misses > 0, not a
                  separate status, so a coasting track that re-associates simply
                  resets its miss counter.
    deleted    -- exceeded the coast limit (confirmed) or failed to confirm
                  (tentative); removed from the bank.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kalman

TENTATIVE = 0
CONFIRMED = 1


@dataclass
class LifecycleParams:
    confirm_hits: int = 3        # hits needed to promote tentative -> confirmed
    confirm_window: int = 5      # ...within this many sweeps of the track's life
    max_coast: int = 5           # confirmed track deleted after this many misses
    max_tentative_miss: int = 2  # tentative track deleted after this many misses
    init_pos_var: float = 1.0e4  # initial position variance (m^2)
    init_vel_var: float = 1.0e4  # initial velocity variance (m^2/s^2)


class TrackBank:
    """Owns all live tracks as stacked arrays plus per-track trajectory history."""

    def __init__(self, params: LifecycleParams | None = None):
        self.p = params or LifecycleParams()
        self._next_id = 1

        self.x = np.zeros((0, kalman.STATE_DIM))
        self.P = np.zeros((0, kalman.STATE_DIM, kalman.STATE_DIM))
        self.ids = np.zeros((0,), dtype=np.int64)
        self.status = np.zeros((0,), dtype=np.int8)
        self.hits = np.zeros((0,), dtype=np.int32)
        self.misses = np.zeros((0,), dtype=np.int32)      # consecutive misses
        self.age = np.zeros((0,), dtype=np.int32)          # sweeps since spawn
        # trajectory history keyed by track id: list of (t, x, y, z)
        self.history: dict[int, list] = {}

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def count(self) -> int:
        return self.x.shape[0]

    def confirmed_mask(self) -> np.ndarray:
        return self.status == CONFIRMED

    # ------------------------------------------------------------------ spawn
    def spawn(self, positions: np.ndarray, t: float) -> None:
        """Create tentative tracks from unassigned ping positions (N, 3)."""
        n = positions.shape[0]
        if n == 0:
            return
        new_x = np.zeros((n, kalman.STATE_DIM))
        new_x[:, :3] = positions
        new_P = np.zeros((n, kalman.STATE_DIM, kalman.STATE_DIM))
        eye3 = np.eye(3)
        new_P[:, :3, :3] = eye3 * self.p.init_pos_var
        new_P[:, 3:, 3:] = eye3 * self.p.init_vel_var

        new_ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n
        for i, tid in enumerate(new_ids):
            self.history[int(tid)] = [(t, *positions[i])]

        self.x = np.vstack([self.x, new_x])
        self.P = np.vstack([self.P, new_P])
        self.ids = np.concatenate([self.ids, new_ids])
        self.status = np.concatenate([self.status, np.full(n, TENTATIVE, np.int8)])
        self.hits = np.concatenate([self.hits, np.ones(n, np.int32)])
        self.misses = np.concatenate([self.misses, np.zeros(n, np.int32)])
        self.age = np.concatenate([self.age, np.zeros(n, np.int32)])

    # ---------------------------------------------------------------- predict
    def predict(self, dt: float, q: float) -> None:
        self.x, self.P = kalman.predict(self.x, self.P, dt, q)
        self.age += 1

    # ----------------------------------------------------------- book-keeping
    def apply_updates(self, track_idx: np.ndarray, meas: np.ndarray,
                      R: np.ndarray, t: float) -> None:
        """Kalman-update the given track indices with matched measurements."""
        if track_idx.size == 0:
            return
        xu, Pu = kalman.update(self.x[track_idx], self.P[track_idx], meas, R)
        self.x[track_idx] = xu
        self.P[track_idx] = Pu
        self.hits[track_idx] += 1
        self.misses[track_idx] = 0
        for j, ti in enumerate(track_idx):
            tid = int(self.ids[ti])
            self.history[tid].append((t, *self.x[ti, :3]))

    def mark_missed(self, track_idx: np.ndarray, t: float) -> None:
        """Flag unassigned tracks as coasting (state already predicted)."""
        if track_idx.size == 0:
            return
        self.misses[track_idx] += 1
        for ti in track_idx:
            tid = int(self.ids[ti])
            self.history[tid].append((t, *self.x[ti, :3]))

    def promote(self) -> None:
        """Promote tentative tracks that have accumulated enough hits."""
        tentative = self.status == TENTATIVE
        ready = tentative & (self.hits >= self.p.confirm_hits) & \
            (self.age < self.p.confirm_window + self.p.confirm_hits)
        self.status[ready] = CONFIRMED

    def prune(self) -> np.ndarray:
        """Delete tracks past their coast/confirmation limits.

        Returns the ids of deleted tracks.
        """
        confirmed = self.status == CONFIRMED
        tentative = self.status == TENTATIVE
        dead = ((confirmed & (self.misses > self.p.max_coast)) |
                (tentative & (self.misses > self.p.max_tentative_miss)))
        dead_ids = self.ids[dead].copy()
        if dead.any():
            keep = ~dead
            self.x = self.x[keep]
            self.P = self.P[keep]
            self.ids = self.ids[keep]
            self.status = self.status[keep]
            self.hits = self.hits[keep]
            self.misses = self.misses[keep]
            self.age = self.age[keep]
        return dead_ids

    # -------------------------------------------------------------- snapshots
    def snapshot(self):
        """Lightweight view of current confirmed + coasting tracks for output."""
        out = []
        for i in range(self.count):
            out.append({
                "id": int(self.ids[i]),
                "status": "confirmed" if self.status[i] == CONFIRMED else "tentative",
                "coasting": bool(self.misses[i] > 0),
                "pos": self.x[i, :3].tolist(),
                "vel": self.x[i, 3:6].tolist(),
                "hits": int(self.hits[i]),
                "misses": int(self.misses[i]),
            })
        return out
