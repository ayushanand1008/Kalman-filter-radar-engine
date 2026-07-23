"""Multi-Target Radar Data Association & Tracking Engine (Python core).

A vectorized numpy/scipy implementation of the MTT pipeline:
geo transforms -> Kalman prediction -> spatial gating -> data association
-> track lifecycle management, with synthetic stress scenarios and metrics.
"""
from . import adsb, association, gating, geo, kalman, metrics, scenarios, track, tracker
from .tracker import Tracker, TrackerConfig
from .track import LifecycleParams

__all__ = [
    "adsb", "association", "gating", "geo", "kalman", "metrics", "scenarios",
    "track", "tracker", "Tracker", "TrackerConfig", "LifecycleParams",
]
__version__ = "0.1.0"
