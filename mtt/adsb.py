"""ADS-B live data integration via the OpenSky Network REST API.

Converts real aircraft state vectors into the metric ENU detection format
expected by `Tracker.step(t, pings)`, with no new runtime dependencies
(urllib + json are stdlib).

Usage
-----
    from mtt.adsb import AirspaceSurveillance

    # Poll 200 km radius around JFK
    sky = AirspaceSurveillance("JFK")
    t, pings, icao_ids = sky.fetch_sweep()
    # pings  : (M, 3) ENU metres  → feed straight to Tracker.step()
    # icao_ids: list[str] of length M  → pseudo ground-truth labels

OpenSky API
-----------
    Anonymous : 1 request / 10 s
    Registered: 1 request /  5 s  (OAuth2 client credentials, optional)

    Set env vars to authenticate:
        OPENSKY_CLIENT_ID=<id>
        OPENSKY_CLIENT_SECRET=<secret>
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geo import LocalFrame

# ---------------------------------------------------------------------------
# Known airport anchors (lat_deg, lon_deg, alt_m)
# ---------------------------------------------------------------------------
KNOWN_AIRPORTS: dict[str, tuple[float, float, float]] = {
    "VNS": (25.4524,  82.8593,  80.0),   # Varanasi, India
    "DEL": (28.5562,  77.1000, 237.0),   # Delhi Indira Gandhi, India
    "JFK": (40.6413, -73.7781,   4.0),   # New York JFK, USA
    "BOM": (19.0896,  72.8656,  14.0),   # Mumbai Chhatrapati Shivaji, India
    "LHR": (51.4700,  -0.4543,  25.0),   # London Heathrow, UK
    "LAX": (33.9425, -118.4081,  38.0),  # Los Angeles, USA
    "DXB": (25.2532,  55.3657,  19.0),   # Dubai International, UAE
    "SYD": (-33.9399, 151.1753,   6.0),  # Sydney Kingsford Smith, Australia
}

_OPENSKY_BASE = "https://opensky-network.org/api"
_TOKEN_URL    = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"

# State-vector field indices in the OpenSky /states/all response
_F_ICAO24     = 0
_F_CALLSIGN   = 1
_F_LON        = 5
_F_LAT        = 6
_F_BARO_ALT   = 7
_F_ON_GROUND  = 8
_F_GEO_ALT    = 13


# ---------------------------------------------------------------------------
# Optional OAuth2 token manager
# ---------------------------------------------------------------------------
@dataclass
class _TokenManager:
    client_id: str
    client_secret: str
    _token: str = field(default="", init=False)
    _expires_at: float = field(default=0.0, init=False)

    def get(self) -> str:
        if time.time() < self._expires_at - 30:
            return self._token
        data = (
            f"grant_type=client_credentials"
            f"&client_id={self.client_id}"
            f"&client_secret={self.client_secret}"
        ).encode()
        req = urllib.request.Request(
            _TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            tok = json.loads(resp.read())
        self._token = tok["access_token"]
        self._expires_at = time.time() + tok.get("expires_in", 3600)
        return self._token


def _make_token_manager() -> Optional[_TokenManager]:
    cid = os.environ.get("OPENSKY_CLIENT_ID", "")
    sec = os.environ.get("OPENSKY_CLIENT_SECRET", "")
    if cid and sec:
        return _TokenManager(client_id=cid, client_secret=sec)
    return None


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------
def fetch_states(
    lamin: float, lomin: float, lamax: float, lomax: float,
    token_mgr: Optional[_TokenManager] = None,
    timeout: float = 20.0,
) -> tuple[float, list]:
    """GET /states/all with bounding box.

    Returns ``(api_time, states)`` where *states* is the raw list from
    OpenSky (may be ``None`` → returned as ``[]``).

    Raises ``urllib.error.URLError`` on network failure so callers can catch
    and degrade gracefully.
    """
    url = (
        f"{_OPENSKY_BASE}/states/all"
        f"?lamin={lamin:.6f}&lomin={lomin:.6f}"
        f"&lamax={lamax:.6f}&lomax={lomax:.6f}"
    )
    headers: dict[str, str] = {"User-Agent": "radar-mtt/0.2"}
    if token_mgr is not None:
        headers["Authorization"] = f"Bearer {token_mgr.get()}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())

    api_time = float(data.get("time", time.time()))
    states   = data.get("states") or []
    return api_time, states


# ---------------------------------------------------------------------------
# Conversion: raw states → ENU pings
# ---------------------------------------------------------------------------
def states_to_pings(
    states: list,
    frame: LocalFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert OpenSky state vectors to ENU detections.

    Parameters
    ----------
    states : raw list of state vectors from the API
    frame  : LocalFrame anchored at the radar/airport origin

    Returns
    -------
    pings      : (M, 3) float64 ENU metres — tracker input
    icao_ids   : list[str] of length M — ICAO24 transponder codes (pseudo ground-truth)
    callsigns  : list[str] of length M — human-readable callsign (may equal icao if blank)

    Aircraft with ``on_ground=True`` or missing lat/lon/alt are silently
    skipped; they never appear in the output arrays.
    """
    pings_list: list[np.ndarray] = []
    icao_ids:   list[str]        = []
    callsigns:  list[str]        = []

    for s in states:
        # --- filter on-ground and missing positions ---
        if s[_F_ON_GROUND]:
            continue
        lon = s[_F_LON]
        lat = s[_F_LAT]
        if lat is None or lon is None:
            continue

        # prefer geometric (GPS) altitude; fall back to barometric; then 0 m
        alt = s[_F_GEO_ALT]
        if alt is None:
            alt = s[_F_BARO_ALT]
        if alt is None:
            alt = 0.0

        enu = frame.geodetic_to_enu(lat, lon, alt)
        pings_list.append(enu)

        icao = str(s[_F_ICAO24] or "")
        cs   = str(s[_F_CALLSIGN] or "").strip() or icao
        icao_ids.append(icao)
        callsigns.append(cs)

    if pings_list:
        pings = np.vstack(pings_list)         # (M, 3)
    else:
        pings = np.empty((0, 3), dtype=float)

    return pings, icao_ids, callsigns


# ---------------------------------------------------------------------------
# High-level surveillance object
# ---------------------------------------------------------------------------
class AirspaceSurveillance:
    """Stateful live-polling wrapper for a circular airspace volume.

    Parameters
    ----------
    airport      : ICAO airport code in KNOWN_AIRPORTS, or ``"custom"``
    lat0, lon0, alt0_m : used when ``airport="custom"``
    radius_km    : horizontal radius of the surveillance volume (km)
    poll_interval: minimum seconds between API calls (honours rate limits)

    Example
    -------
    ::

        sky = AirspaceSurveillance("JFK", radius_km=200)
        for t, pings, icao_ids, callsigns in sky.iter_sweeps(n=20):
            report = tracker.step(t, pings)
    """

    def __init__(
        self,
        airport: str = "JFK",
        *,
        lat0: float = 0.0,
        lon0: float = 0.0,
        alt0_m: float = 0.0,
        radius_km: float = 200.0,
        poll_interval: float = 10.0,   # seconds; 10 = anonymous limit
    ):
        if airport.upper() != "CUSTOM":
            key = airport.upper()
            if key not in KNOWN_AIRPORTS:
                raise ValueError(
                    f"Unknown airport {airport!r}. "
                    f"Choose from {list(KNOWN_AIRPORTS)} or use airport='custom' "
                    f"with explicit lat0/lon0/alt0_m."
                )
            lat0, lon0, alt0_m = KNOWN_AIRPORTS[key]
        self.frame   = LocalFrame(lat0, lon0, alt0_m)
        self.origin  = (lat0, lon0, alt0_m)
        self.airport = airport.upper()
        self.radius_km     = float(radius_km)
        self.poll_interval = float(poll_interval)
        self._token_mgr    = _make_token_manager()
        self._last_poll    = 0.0

        # bounding box in degrees (approx — lat deg ≈ 111 km everywhere,
        # lon deg shrinks with cos(lat) but we stay conservative)
        deg_lat = radius_km / 111.0
        deg_lon = radius_km / (111.0 * max(abs(np.cos(np.deg2rad(lat0))), 0.01))
        self._bbox = (
            lat0 - deg_lat, lon0 - deg_lon,
            lat0 + deg_lat, lon0 + deg_lon,
        )

    # ------------------------------------------------------------------
    def fetch_sweep(
        self,
    ) -> tuple[float, np.ndarray, list[str], list[str]]:
        """Fetch one live sweep, honouring the poll-interval rate limit.

        Blocks (``time.sleep``) if called faster than ``poll_interval``.

        Returns ``(t, pings, icao_ids, callsigns)``.
        """
        wait = self._last_poll + self.poll_interval - time.time()
        if wait > 0:
            time.sleep(wait)

        lamin, lomin, lamax, lomax = self._bbox
        api_t, states = fetch_states(lamin, lomin, lamax, lomax, self._token_mgr)
        self._last_poll = time.time()

        pings, icao_ids, callsigns = states_to_pings(states, self.frame)

        # clip to true circular radius (bounding box is a superset)
        if pings.shape[0]:
            r = np.linalg.norm(pings[:, :2], axis=1)   # horizontal distance
            mask = r <= self.radius_km * 1000.0
            pings     = pings[mask]
            icao_ids  = [icao_ids[i]  for i, ok in enumerate(mask) if ok]
            callsigns = [callsigns[i] for i, ok in enumerate(mask) if ok]

        return api_t, pings, icao_ids, callsigns

    # ------------------------------------------------------------------
    def iter_sweeps(self, n: int = 0):
        """Yield ``(t, pings, icao_ids, callsigns)`` for *n* sweeps.

        If ``n <= 0``, yields indefinitely until interrupted.
        """
        count = 0
        while n <= 0 or count < n:
            yield self.fetch_sweep()
            count += 1

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        lat0, lon0, _ = self.origin
        return (
            f"AirspaceSurveillance(airport={self.airport!r}, "
            f"origin=({lat0:.4f}, {lon0:.4f}), "
            f"radius={self.radius_km:.0f} km)"
        )
