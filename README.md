# Multi-Target Radar Data Association & Tracking Engine (MTT)

A high-performance, real-time Multi-Target Tracking (MTT) engine designed to ingest raw, noisy spatial detections and associate them into clean, continuous trajectories. The engine supports both complex synthetic radar scenarios and live integration with real-world ADS-B data via the OpenSky Network.

## Language & Architecture Choices

The core engine is built entirely in **Python** using **NumPy** and **SciPy**. 

While C++ is traditional for radar tracking, Python + NumPy was chosen because it allows for a highly optimized **Vectorized Structure-of-Arrays (SoA)** architecture. Instead of looping over individual aircraft objects, the entire state of the sky is maintained as large continuous matrices. We process 10,000 aircraft tracks simultaneously through a single BLAS/LAPACK matrix operation. This hybrid approach delivers near C-level performance (sub-millisecond per-sweep latency, processing tens of thousands of detections per second) while maintaining Python's readability, flexibility, and rapid iteration speed.

---

## The Tracking Pipeline: Algorithms & Mathematics

The engine processes data in a strict 6-stage pipeline.

### 1. Geodesy & Coordinate Transformation (WGS84 → ECEF → ENU)
Radar and ADS-B data arrive in spherical geographic coordinates (Latitude, Longitude, Altitude). 
**Why:** Standard Kalman Filters require linear Cartesian dynamics to model objects moving in straight lines. Using curved spherical coordinates would require highly non-linear Extended Kalman Filters (EKF) with expensive Jacobians.
**Algorithm:** We convert the WGS84 geographic data into Earth-Centered Earth-Fixed (ECEF) 3D coordinates, and then rotate them into a local East-North-Up (ENU) tangent plane centered on the radar site. This creates a flat 3D Cartesian grid where constant-velocity math perfectly holds true.

### 2. State Estimation (Constant-Velocity Kalman Filter)
**Why:** Raw radar measurements contain Gaussian noise and only provide position, not speed. 
**Algorithm:** We use a linear Kalman Filter with a 6-dimensional state vector `[x, y, z, vx, vy, vz]`. The filter operates in two steps:
* **Predict:** Extrapolates the current position into the future based on velocity and adds process noise (representing expected pilot maneuvers or wind).
* **Update (Joseph Form):** Fuses the noisy incoming radar measurement with the prediction. The Joseph form equation is used because it guarantees the covariance matrix remains mathematically positive semi-definite and stable during floating-point operations.

### 3. Spatial Indexing (cKDTree)
**Why:** Comparing every track to every raw radar ping requires $O(N \times M)$ distance calculations. At 10,000 tracks, this becomes a massive bottleneck.
**Algorithm:** We dump all raw pings into a SciPy `cKDTree` (K-Dimensional Tree). This reduces spatial queries to $O(N \log M)$, allowing the engine to instantly pull only the pings that fall within a rough geographic bounding box of a track.

### 4. Statistical Gating (Mahalanobis Distance)
**Why:** Simple Euclidean distance is flawed because it ignores velocity and uncertainty. An aircraft flying at Mach 1 has a very long, narrow ellipse of where it could possibly be next.
**Algorithm:** We compute the **Mahalanobis Distance** using the Kalman filter's innovation covariance matrix ($S$). This creates a multi-dimensional statistical gate (an ellipse angled in the direction of flight) that instantly prunes any radar pings that are physically impossible to reach, preventing mis-associations.

### 5. Data Association (Sub-Graph Hungarian Assignment)
**Why:** When two aircraft cross paths closely, greedy "nearest-neighbor" matching often steals the wrong ping and causes an ID switch. 
**Algorithm:** We use the **Hungarian Algorithm (Munkres)**, which finds the globally optimal assignment that minimizes the total cost across all tracks simultaneously. The cost function is the **Gaussian Negative Log-Likelihood**, mathematically preferring matches that closely align with the Kalman covariance. 
To avoid the intense $O(N^3)$ computational cost of the Hungarian algorithm, we use **Sub-Graph Fracturing**. We split the sky into tiny, isolated clusters of interacting aircraft and run the Hungarian algorithm independently on each micro-cluster.

### 6. Track Lifecycle Management (M-of-N Logic)
**Why:** Not all radar pings are real; birds, weather, and ground clutter cause false alarms. Similarly, a real aircraft might be obscured by a mountain for a sweep.
**Algorithm:** We use heuristic lifecycle state machines:
* **Tentative:** A new track requires multiple hits across consecutive sweeps to prove it isn't random noise.
* **Confirmed:** Once proven, it is handed off to the user.
* **Coasting:** If a confirmed track misses a radar ping, it survives via Kalman prediction for several sweeps before eventually being pruned as **Deleted**.

---

## Frontend Radar UI Guide

The system includes a zero-dependency web frontend to visualize the engine tracking data in real-time.

### Controls & Options (Sidebar)
* **Synthetic Scenarios:** Play back offline stress tests (Varanasi, Delhi, JFK) with hundreds of targets to visualize complex crossovers and holding patterns.
* **High-Speed Playback:** Play back recorded 20-minute chunks of real-world data (LATEST or STORED).
* **Live ADS-B:** Connect directly to the OpenSky network to track real, live flights happening right now around 8 major global hubs.
* **Feed Speed:** Adjust the playback rate from 1 to 30 frames per second (Disabled during Live Mode, which updates according to API limits).
* **Live State Panel:** Real-time metrics showing how many tracks the engine is spawning, tracking, and deleting per sweep.
* **DVR & Snapshot Recording:** The system continuously buffers a rolling 20-minute window of whatever airport you are viewing. You can click **Record New Snapshot** to save a permanent 20-minute JSON file of the airspace to your hard drive for later high-speed playback.

### Map & Legend
* **The Radar Scope:** The map is a 2D projection of the 3D local ENU plane. Concentric grey rings represent distance ranges (every 10km) from the radar origin at the center.

**Symbology:**
* **Dark Green Dots (Pings):** Raw, unfiltered radar detections or ground clutter.
* **Grey/Blue Rings (Tentative):** Newly spotted objects. The system is tracking them internally but waiting to verify they are real aircraft before fully rendering them.
* **Cyan Boxes with Tails (Confirmed):** Verified aircraft tracks. 
  * A trailing line shows the recent path history.
  * A forward-pointing vector shows the predicted velocity/heading.
  * Labels indicate the aircraft Callsign and its exact Altitude (in km).
* **Yellow Boxes with "CST" (Coasting):** An aircraft that the radar temporarily lost sight of. The engine is "flying blind" and predicting its location using physics until it reappears.
