"""Synthetic radar scenario generators with ground truth.

Replaces the spec's PostgreSQL/PostGIS ADS-B corpus. Each scenario produces:
  * a LocalFrame anchored at the airport,
  * ground-truth trajectories (per-target ENU state over time),
  * per-sweep detections with realistic measurement noise, missed detections,
    and false alarms -- in both ENU (engine input) and WGS84 (feed/frontend).

Because we synthesize the truth, we can score the tracker (see metrics.py) --
something a real, unlabeled feed could not give us.

Three stress profiles from the spec:
  VNS  - sparse straight approaches (basic lifecycle)
  DEL  - dense coordinated-turn holding patterns (non-linear kinematics)
  JFK  - heavy straight-line crossovers (assignment / gating stress)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geo import LocalFrame

# (lat, lon, elevation_m) of each anchor airport.
AIRPORTS = {
    "VNS": (25.4524, 82.8593, 80.0),
    "DEL": (28.5562, 77.1000, 237.0),
    "JFK": (40.6413, -73.7781, 4.0),
}


@dataclass
class Scenario:
    name: str
    frame: LocalFrame
    times: np.ndarray                       # (T,)
    sweeps_enu: list                        # list of (t, (M,3) enu)
    sweeps_geo: list                        # list of (t, (M,3) lat/lon/alt)
    truth: dict                             # target_id -> (T,3) enu (nan if absent)
    meta: dict = field(default_factory=dict)


def _cv_target(rng, p0, v0, times):
    """Constant-velocity ground truth: (T,3)."""
    return p0[None, :] + times[:, None] * v0[None, :]


def _turn_target(rng, center, radius, omega, phase, alt, times):
    """Coordinated-turn (level circular) ground truth: (T,3)."""
    ang = omega * times + phase
    x = center[0] + radius * np.cos(ang)
    y = center[1] + radius * np.sin(ang)
    z = np.full_like(times, alt)
    return np.stack([x, y, z], axis=-1)


def _build_sweeps(rng, truth, times, meas_std, p_miss, fa_rate, extent):
    """Turn ground-truth trajectories into noisy detection sweeps."""
    sweeps = []
    for ti, t in enumerate(times):
        dets = []
        for traj in truth.values():
            p = traj[ti]
            if np.any(np.isnan(p)):
                continue
            if rng.random() < p_miss:
                continue                      # missed detection
            dets.append(p + rng.normal(0.0, meas_std, size=3))
        # false alarms: uniform clutter in the horizontal extent, plausible alt
        n_fa = rng.poisson(fa_rate)
        for _ in range(n_fa):
            fa = np.array([
                rng.uniform(-extent, extent),
                rng.uniform(-extent, extent),
                rng.uniform(500.0, 11000.0),
            ])
            dets.append(fa)
        arr = np.asarray(dets, dtype=float).reshape(-1, 3)
        # shuffle so detection order carries no identity information
        rng.shuffle(arr)
        sweeps.append((float(t), arr))
    return sweeps


# Recommended process-noise PSD per scenario. Turning targets (DEL) need a
# looser CV model (higher q) to follow maneuvers; straight-line scenes stay
# tight. This is the single-model tradeoff IMM is designed to remove.
RECOMMENDED_Q = {"VNS": 3.0, "DEL": 150.0, "JFK": 8.0}


def _finalize(name, origin, truth, times, meas_std, p_miss, fa_rate, extent, rng):
    frame = LocalFrame(*origin)
    sweeps_enu = _build_sweeps(rng, truth, times, meas_std, p_miss, fa_rate, extent)
    sweeps_geo = []
    for t, enu in sweeps_enu:
        if len(enu):
            lat, lon, alt = frame.enu_to_geodetic(enu)
            geo = np.stack([lat, lon, alt], axis=-1)
        else:
            geo = enu.reshape(-1, 3)
        sweeps_geo.append((t, geo))
    return Scenario(
        name=name, frame=frame, times=times,
        sweeps_enu=sweeps_enu, sweeps_geo=sweeps_geo, truth=truth,
        meta={"meas_std": meas_std, "p_miss": p_miss,
              "fa_rate": fa_rate, "n_targets": len(truth),
              "origin": origin, "q_hint": RECOMMENDED_Q.get(name, 5.0)},
    )


def varanasi(seed=0):
    """VNS - sparse straight approaches. 4 aircraft, light clutter."""
    rng = np.random.default_rng(seed)
    times = np.arange(0, 60, dtype=float)
    truth = {}
    # inbound aircraft on descending approaches from different bearings
    starts = [
        (np.array([-40000.,  30000., 6000.]), np.array([250., -180., -60.])),
        (np.array([ 35000.,  38000., 6500.]), np.array([-230., -210., -70.])),
        (np.array([-30000., -35000., 5500.]), np.array([210.,  230., -55.])),
        (np.array([ 42000., -20000., 7000.]), np.array([-260.,  140., -75.])),
    ]
    for i, (p0, v0) in enumerate(starts):
        truth[i] = _cv_target(rng, p0, v0, times)
    return _finalize("VNS", AIRPORTS["VNS"], truth, times,
                     meas_std=30.0, p_miss=0.05, fa_rate=2.0,
                     extent=60000.0, rng=rng)


def delhi(seed=1):
    """DEL - dense holding patterns. 14 aircraft in coordinated turns."""
    rng = np.random.default_rng(seed)
    times = np.arange(0, 90, dtype=float)
    truth = {}
    n = 14
    for i in range(n):
        center = np.array([rng.uniform(-25000, 25000),
                           rng.uniform(-25000, 25000)])
        radius = rng.uniform(4000, 9000)
        omega = rng.choice([-1, 1]) * rng.uniform(0.02, 0.05)  # rad/s
        phase = rng.uniform(0, 2 * np.pi)
        alt = rng.uniform(3000, 9000)
        truth[i] = _turn_target(rng, center, radius, omega, phase, alt, times)
    return _finalize("DEL", AIRPORTS["DEL"], truth, times,
                     meas_std=35.0, p_miss=0.10, fa_rate=12.0,
                     extent=45000.0, rng=rng)


def jfk(seed=2):
    """JFK - extreme crossovers. 36 aircraft on straight intersecting paths."""
    rng = np.random.default_rng(seed)
    times = np.arange(0, 100, dtype=float)
    truth = {}
    n = 36
    for i in range(n):
        bearing = rng.uniform(0, 2 * np.pi)
        # start far out on the bearing, fly through the center region
        d0 = rng.uniform(35000, 55000)
        p0 = np.array([d0 * np.cos(bearing), d0 * np.sin(bearing),
                       rng.uniform(2500, 10500)])
        speed = rng.uniform(200, 280)
        # velocity aimed roughly at the origin with lateral jitter -> crossings
        aim = -p0[:2] / np.linalg.norm(p0[:2])
        jitter = rng.normal(0, 0.15, size=2)
        vdir = aim + jitter
        vdir = vdir / np.linalg.norm(vdir)
        v0 = np.array([vdir[0] * speed, vdir[1] * speed, rng.uniform(-20, 20)])
        truth[i] = _cv_target(rng, p0, v0, times)
    return _finalize("JFK", AIRPORTS["JFK"], truth, times,
                     meas_std=40.0, p_miss=0.08, fa_rate=30.0,
                     extent=60000.0, rng=rng)


GENERATORS = {"VNS": varanasi, "DEL": delhi, "JFK": jfk}


def generate(name: str, seed: int | None = None) -> Scenario:
    name = name.upper()
    if name not in GENERATORS:
        raise ValueError(f"unknown scenario {name!r}; choose from {list(GENERATORS)}")
    gen = GENERATORS[name]
    return gen() if seed is None else gen(seed)
