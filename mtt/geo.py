"""Geospatial transforms: WGS84 <-> ECEF <-> ENU (Local Tangent Plane).

The tracking engine runs in a flat, metric ENU frame anchored at a radar
origin. Raw detections arrive as geodetic (lat, lon, alt) and are converted to
ENU before gating/filtering; confirmed states are converted back to WGS84
before broadcasting so the frontend map stays aligned.

All functions are fully vectorized: pass scalars or numpy arrays of shape
(..., 3) and get the same batch shape back.
"""
from __future__ import annotations

import numpy as np

# WGS84 ellipsoid constants
_A = 6378137.0                      # semi-major axis (m)
_F = 1.0 / 298.257223563           # flattening
_E2 = _F * (2.0 - _F)              # first eccentricity squared
_B = _A * (1.0 - _F)               # semi-minor axis
_EP2 = (_A * _A - _B * _B) / (_B * _B)  # second eccentricity squared


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    """(lat, lon, alt) in deg/deg/m -> ECEF (X, Y, Z) in meters."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    h = np.asarray(alt_m, dtype=float)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    n = _A / np.sqrt(1.0 - _E2 * sin_lat * sin_lat)
    x = (n + h) * cos_lat * cos_lon
    y = (n + h) * cos_lat * sin_lon
    z = (n * (1.0 - _E2) + h) * sin_lat
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(ecef):
    """ECEF (..., 3) in meters -> (lat_deg, lon_deg, alt_m) via Bowring's method."""
    ecef = np.asarray(ecef, dtype=float)
    x = ecef[..., 0]
    y = ecef[..., 1]
    z = ecef[..., 2]

    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    # Bowring's initial parametric latitude
    theta = np.arctan2(z * _A, p * _B)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    lat = np.arctan2(z + _EP2 * _B * sin_t ** 3,
                     p - _E2 * _A * cos_t ** 3)
    sin_lat = np.sin(lat)
    n = _A / np.sqrt(1.0 - _E2 * sin_lat * sin_lat)
    # Altitude; guard the polar singularity where p -> 0
    alt = np.where(np.abs(np.cos(lat)) > 1e-12,
                   p / np.cos(lat) - n,
                   z / np.sin(lat) - n * (1.0 - _E2))
    return np.rad2deg(lat), np.rad2deg(lon), alt


def _enu_rotation(lat0_deg, lon0_deg):
    """Rotation matrix mapping an ECEF *offset* vector into ENU at the origin."""
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    sl, cl = np.sin(lat0), np.cos(lat0)
    so, co = np.sin(lon0), np.cos(lon0)
    return np.array([
        [-so,        co,       0.0],
        [-sl * co,  -sl * so,  cl],
        [ cl * co,   cl * so,  sl],
    ])


class LocalFrame:
    """A Local Tangent Plane (ENU) anchored at a geodetic origin.

    Precomputes the origin ECEF vector and rotation matrix so per-sweep
    conversions are a single vectorized matmul.
    """

    def __init__(self, lat0_deg: float, lon0_deg: float, alt0_m: float = 0.0):
        self.lat0 = float(lat0_deg)
        self.lon0 = float(lon0_deg)
        self.alt0 = float(alt0_m)
        self._ecef0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
        self._R = _enu_rotation(lat0_deg, lon0_deg)   # ECEF-offset -> ENU
        self._Rt = self._R.T                          # ENU -> ECEF-offset

    def geodetic_to_enu(self, lat_deg, lon_deg, alt_m):
        """(lat, lon, alt) -> ENU (..., 3) in meters."""
        ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        offset = ecef - self._ecef0
        return offset @ self._Rt  # (..., 3) @ (3,3) applies R to each row

    def enu_to_geodetic(self, enu):
        """ENU (..., 3) -> (lat_deg, lon_deg, alt_m)."""
        enu = np.asarray(enu, dtype=float)
        ecef = enu @ self._R + self._ecef0  # inverse rotation is R^T applied on the right
        return ecef_to_geodetic(ecef)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"LocalFrame(lat0={self.lat0:.5f}, lon0={self.lon0:.5f}, alt0={self.alt0:.1f})"
