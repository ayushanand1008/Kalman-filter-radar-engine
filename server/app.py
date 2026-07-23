#!/usr/bin/env python3
"""Zero-dependency streaming server for the MTT engine.

Uses only the Python standard library. The spec calls for a WebSocket feed, but
a radar feed is one-way (server -> client), so Server-Sent Events over plain
HTTP is a simpler, dependency-free fit. The client picks a scenario via a query
parameter; the server replays that scenario's sweeps through the live tracker
and streams each frame's detections + confirmed track states as they are
produced -- simulating a real-time radar feed.

Routes
------
    GET /                                   -> the Canvas radar-scope frontend
    GET /scenarios                          -> JSON list of scenarios + metadata
    GET /metrics?scenario=JFK               -> JSON validation scorecard
    GET /status                             -> DVR / Recording status
    POST /record                            -> Trigger 10-min background recording
    GET /stream?scenario=JFK&speed=8        -> SSE stream (synthetic scenario)
    GET /stream?scenario=LIVE&airport=JFK   -> SSE stream (live OpenSky ADS-B)
    GET /stream?scenario=LATEST_JFK         -> SSE stream (DVR playback)
    GET /stream?scenario=STORED_JFK         -> SSE stream (Stored playback)

Run:  python -m server.app   (from the radar_mtt directory)
"""
from __future__ import annotations

import json
import os
import sys
import time
import glob
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# allow running as `python server/app.py` or `python -m server.app`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtt import scenarios
from mtt.adsb import KNOWN_AIRPORTS, AirspaceSurveillance
from mtt.metrics import evaluate
from mtt.tracker import Tracker, TrackerConfig

FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "frontend", "index.html")

class DataManager:
    """Runs a background thread polling OpenSky for the DVR buffer and snapshot recording."""
    def __init__(self, airport="JFK"):
        self.airport = airport
        self.sky = AirspaceSurveillance(airport, poll_interval=10.0)
        self.dvr_buffer = deque(maxlen=120) # 20 mins (120 frames at 10s intervals)
        self.recording = False
        self.record_frames = []
        self.record_target = 120
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_target(self, airport: str):
        airport = airport.upper()
        if airport not in KNOWN_AIRPORTS:
            return
        if self.airport != airport:
            print(f"Switching background DVR target to {airport}")
            self.airport = airport
            self.sky = AirspaceSurveillance(airport, poll_interval=10.0)
            self.dvr_buffer.clear()
            self.recording = False
            self.record_frames = []

    def _loop(self):
        import urllib.error
        while True:
            try:
                t, pings, icao_ids, callsigns = self.sky.fetch_sweep()
            except urllib.error.URLError:
                time.sleep(self.sky.poll_interval)
                continue
            except Exception as e:
                print(f"Error in background poll: {e}")
                time.sleep(self.sky.poll_interval)
                continue

            frame_data = {
                "t": t,
                "pings": pings.tolist(),
                "icao_ids": icao_ids,
                "callsigns": callsigns
            }
            self.dvr_buffer.append(frame_data)
            
            if self.recording:
                self.record_frames.append(frame_data)
                if len(self.record_frames) >= self.record_target:
                    self._save_recording()
                    self.recording = False

    def trigger_recording(self, frames=120):
        if not self.recording:
            self.record_frames = []
            self.record_target = frames
            self.recording = True
            return True
        return False

    def _save_recording(self):
        out = {
            "airport": self.airport,
            "origin": list(self.sky.origin),
            "radius_km": self.sky.radius_km,
            "frames": self.record_frames
        }
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", f"stored_{self.airport.lower()}.json")
        try:
            with open(path, "w") as f:
                json.dump(out, f)
            print(f"Recording saved to {path}")
        except Exception as e:
            print(f"Failed to save recording: {e}")

# Global instance for DVR/Recording
DATA_MGR = DataManager("JFK")


def _scenario_list():
    out = []
    # 1. Synthetic scenarios
    for name, gen in scenarios.GENERATORS.items():
        scn = gen()
        out.append({
            "name": name,
            "origin": scn.meta["origin"],
            "n_targets": scn.meta["n_targets"],
            "n_frames": len(scn.sweeps_enu),
            "meas_std": scn.meta["meas_std"],
            "live": False,
            "description": (gen.__doc__ or "").splitlines()[0].strip(),
        })
        
    code = DATA_MGR.airport
    lat, lon, alt = KNOWN_AIRPORTS[code]
    
    # 2. DVR Latest (dynamically tracks DataManager's current target)
    out.append({
        "name": f"LATEST_{code}",
        "origin": [lat, lon, alt],
        "n_targets": None,
        "n_frames": len(DATA_MGR.dvr_buffer),
        "meas_std": 50.0,
        "live": False,
        "airport": code,
        "description": f"Latest DVR Buffer ({code})",
    })
    
    # 3. Stored Snapshots (scan the data/ folder for any previously saved snapshot)
    for path in glob.glob(os.path.join("data", "stored_*.json")):
        try:
            acode = os.path.basename(path).replace("stored_", "").replace(".json", "").upper()
            if acode in KNOWN_AIRPORTS:
                alat, alon, aalt = KNOWN_AIRPORTS[acode]
                out.append({
                    "name": f"STORED_{acode}",
                    "origin": [alat, alon, aalt],
                    "n_targets": None,
                    "n_frames": 120,
                    "meas_std": 50.0,
                    "live": False,
                    "airport": acode,
                    "description": f"Stored 20-min Snapshot ({acode})",
                })
        except Exception:
            pass
            
    # 4. Live feeds
    for acode, (alat, alon, aalt) in KNOWN_AIRPORTS.items():
        out.append({
            "name": f"LIVE_{acode}",
            "origin": [alat, alon, aalt],
            "n_targets": None,
            "n_frames": None,
            "meas_std": 50.0,
            "live": True,
            "airport": acode,
            "description": f"Live ADS-B · {acode} · OpenSky Network",
        })
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/record":
            success = DATA_MGR.trigger_recording()
            self._send_json({"success": success})
        else:
            self.send_error(404, "not found")

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if route == "/" or route == "/index.html":
                self._serve_frontend()
            elif route == "/scenarios":
                self._send_json(_scenario_list())
            elif route == "/status":
                target = qs.get("airport", [DATA_MGR.airport])[0].upper()
                if target in KNOWN_AIRPORTS:
                    DATA_MGR.set_target(target)
                    
                path = os.path.join("data", f"stored_{DATA_MGR.airport.lower()}.json")
                self._send_json({
                    "airport": DATA_MGR.airport,
                    "dvr_frames": len(DATA_MGR.dvr_buffer),
                    "dvr_max": 120,
                    "recording": DATA_MGR.recording,
                    "rec_frames": len(DATA_MGR.record_frames),
                    "rec_max": DATA_MGR.record_target,
                    "has_stored": os.path.exists(path)
                })
            elif route == "/metrics":
                self._serve_metrics(qs)
            elif route == "/stream":
                name = qs.get("scenario", ["VNS"])[0].upper()
                if name.startswith("LIVE"):
                    airport = qs.get("airport", [name.replace("LIVE_", "") or "JFK"])[0].upper()
                    self._serve_stream_live(qs, airport)
                elif name.startswith("LATEST_") or name.startswith("STORED_"):
                    self._serve_stream_playback(qs, name)
                else:
                    self._serve_stream(qs)
            else:
                self.send_error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_frontend(self):
        with open(FRONTEND, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_metrics(self, qs):
        name = qs.get("scenario", ["VNS"])[0].upper()
        scn = scenarios.generate(name)
        trk = Tracker.for_scenario(scn)
        reports = [trk.step(t, pings) for t, pings in scn.sweeps_enu]
        self._send_json(evaluate(scn, reports))

    def _serve_stream(self, qs):
        name = qs.get("scenario", ["VNS"])[0].upper()
        speed = float(qs.get("speed", ["8"])[0])
        speed = max(0.5, min(speed, 60.0))
        scn = scenarios.generate(name)
        trk = Tracker.for_scenario(scn)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        self._sse("meta", {
            "scenario": scn.name,
            "origin": scn.meta["origin"],
            "n_targets": scn.meta["n_targets"],
            "n_frames": len(scn.sweeps_enu),
            "live": False,
        })

        interval = 1.0 / speed
        for i, (t, pings) in enumerate(scn.sweeps_enu):
            rep = trk.step(t, pings)
            payload = {
                "frame": rep["frame"],
                "t": rep["t"],
                "pings": pings.round(1).tolist(),
                "tracks": [
                    {"id": tr["id"], "status": tr["status"],
                     "coasting": tr["coasting"],
                     "pos": [round(v, 1) for v in tr["pos"]],
                     "vel": [round(v, 2) for v in tr["vel"]],
                     "alt_km": round(tr["pos"][2] / 1000.0, 2),
                     "label": None}
                    for tr in rep["tracks"]
                ],
                "stats": {
                    "n_pings": rep["n_pings"],
                    "n_confirmed": rep["n_confirmed"],
                    "n_tracks": rep["n_tracks"],
                    "n_spawned": rep["n_spawned"],
                    "n_deleted": rep["n_deleted"],
                },
            }
            self._sse("frame", payload)
            time.sleep(interval)
        self._sse("end", {"scenario": scn.name})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _serve_stream_playback(self, qs, name):
        import numpy as np
        speed = float(qs.get("speed", ["8"])[0])
        speed = max(0.5, min(speed, 60.0))
        
        frames = []
        origin = None
        if name.startswith("LATEST_"):
            frames = list(DATA_MGR.dvr_buffer)
            origin = list(DATA_MGR.sky.origin)
        elif name.startswith("STORED_"):
            code = name.replace("STORED_", "").lower()
            path = os.path.join("data", f"stored_{code}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                    frames = data["frames"]
                    origin = data["origin"]
            else:
                self._send_json({"error": "Stored file not found"}, code=404)
                return

        cfg = TrackerConfig(meas_noise_std=50.0, process_noise_q=8.0)
        trk = Tracker(cfg)
        
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        self._sse("meta", {
            "scenario": name,
            "origin": origin,
            "n_targets": None,
            "n_frames": len(frames),
            "live": False,
        })
        
        id_to_label: dict[int, str] = {}
        interval = 1.0 / speed
        
        for i, frame_data in enumerate(frames):
            t = frame_data["t"]
            pings = np.array(frame_data["pings"])
            icao_ids = frame_data["icao_ids"]
            callsigns = frame_data["callsigns"]
            
            rep = trk.step(t, pings)
            
            confirmed_ids = {tr["id"] for tr in rep["tracks"] if tr["status"] == "confirmed"}
            if pings.shape[0] and confirmed_ids:
                conf_tracks = [tr for tr in rep["tracks"] if tr["id"] in confirmed_ids]
                for tr in conf_tracks:
                    if tr["id"] not in id_to_label:
                        pos = np.array(tr["pos"])
                        dists = np.linalg.norm(pings - pos, axis=1)
                        closest = int(np.argmin(dists))
                        if dists[closest] < 5000 and closest < len(callsigns):
                            id_to_label[tr["id"]] = callsigns[closest]
                            
            payload = {
                "frame": rep["frame"],
                "t": t,
                "pings": pings.round(1).tolist() if pings.shape[0] else [],
                "tracks": [
                    {"id": tr["id"], "status": tr["status"],
                     "coasting": tr["coasting"],
                     "pos": [round(v, 1) for v in tr["pos"]],
                     "vel": [round(v, 2) for v in tr["vel"]],
                     "alt_km": round(tr["pos"][2] / 1000.0, 2) if len(tr["pos"]) > 2 else 0.0,
                     "label": id_to_label.get(tr["id"])}
                    for tr in rep["tracks"]
                ],
                "stats": {
                    "n_pings": rep["n_pings"],
                    "n_confirmed": rep["n_confirmed"],
                    "n_tracks": rep["n_tracks"],
                    "n_spawned": rep["n_spawned"],
                    "n_deleted": rep["n_deleted"],
                },
                "live": False,
                "sweep": i + 1,
            }
            self._sse("frame", payload)
            time.sleep(interval)
            
        self._sse("end", {"scenario": name})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _serve_stream_live(self, qs, airport: str):
        import urllib.error
        radius_km = float(qs.get("radius_km", ["200"])[0])

        try:
            sky = AirspaceSurveillance(airport, radius_km=radius_km, poll_interval=10.0)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, code=400)
            return

        cfg = TrackerConfig(meas_noise_std=50.0, process_noise_q=8.0)
        trk = Tracker(cfg)
        id_to_label: dict[int, str] = {}
        
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        self._sse("meta", {
            "scenario": f"LIVE_{airport}",
            "origin": list(sky.origin),
            "n_targets": None,
            "n_frames": None,
            "live": True,
            "airport": airport,
            "radius_km": radius_km,
        })

        import numpy as np
        sweep_num = 0
        while True:
            try:
                t, pings, icao_ids, callsigns = sky.fetch_sweep()
            except urllib.error.URLError as exc:
                self._sse("error", {"msg": str(exc)})
                time.sleep(sky.poll_interval)
                continue

            rep = trk.step(t, pings)
            sweep_num += 1

            confirmed_ids = {tr["id"] for tr in rep["tracks"] if tr["status"] == "confirmed"}
            if pings.shape[0] and confirmed_ids:
                conf_tracks = [tr for tr in rep["tracks"] if tr["id"] in confirmed_ids]
                for tr in conf_tracks:
                    if tr["id"] in id_to_label:
                        continue
                    pos = np.array(tr["pos"])
                    dists = np.linalg.norm(pings - pos, axis=1)
                    closest = int(np.argmin(dists))
                    if dists[closest] < 5000 and closest < len(callsigns):
                        id_to_label[tr["id"]] = callsigns[closest]

            payload = {
                "frame": rep["frame"],
                "t": t,
                "pings": pings.round(1).tolist(),
                "tracks": [
                    {"id": tr["id"], "status": tr["status"],
                     "coasting": tr["coasting"],
                     "pos": [round(v, 1) for v in tr["pos"]],
                     "vel": [round(v, 2) for v in tr["vel"]],
                     "alt_km": round(tr["pos"][2] / 1000.0, 2),
                     "label": id_to_label.get(tr["id"])}
                    for tr in rep["tracks"]
                ],
                "stats": {
                    "n_pings": rep["n_pings"],
                    "n_confirmed": rep["n_confirmed"],
                    "n_tracks": rep["n_tracks"],
                    "n_spawned": rep["n_spawned"],
                    "n_deleted": rep["n_deleted"],
                },
                "live": True,
                "sweep": sweep_num,
            }
            self._sse("frame", payload)

    def _sse(self, event, data):
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
        self.wfile.write(f"{len(msg):X}\r\n".encode() + msg + b"\r\n")
        self.wfile.flush()

def main():
    host = os.environ.get("MTT_HOST", "127.0.0.1")
    port = int(os.environ.get("MTT_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"MTT engine serving on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()

if __name__ == "__main__":
    main()
