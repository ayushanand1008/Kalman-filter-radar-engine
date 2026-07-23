# User Guide: Multi-Target Radar Tracking Engine

Welcome to the Multi-Target Radar Tracking Engine! This system allows you to simulate and view complex radar scenarios as well as connect directly to the OpenSky Network to track live, real-world flights.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ayushanand1008/Kalman-filter-radar-engine.git
   cd Kalman-filter-radar-engine
   ```

2. **Install the dependencies:**
   The backend relies on `numpy` and `scipy` for high-performance matrix math.
   ```bash
   pip install -r requirements.txt
   ```

## Running the Engine

### Option A: 1-Click Launch (Windows)
If you are on Windows, simply double-click the `start_radar.bat` file in the main directory. This will automatically start the background server and open your default web browser to the correct page.

### Option B: Manual Launch (Mac / Linux / Windows)
1. Open a terminal and run the backend server:
   ```bash
   python -m server.app
   ```
2. Open your web browser and go to:
   ```
   http://127.0.0.1:8000
   ```

## How to use the Web Interface

Once the interface loads, you will see the **Radar Scope** in the center and the **Control Panel** on the right.

### 1. Choosing a Scenario
The sidebar lists several different feeds you can watch:
* **Synthetic:** These are offline stress-tests simulating hundreds of targets performing complex crossovers. They are great for testing the limits of the Kalman Filter and the Hungarian Association algorithm.
* **High-Speed Playback:** If you have recorded a snapshot of a real-world airport, you can play it back here. The repository comes with a 20-minute snapshot of JFK airport (`STORED_JFK`) pre-loaded.
* **Live ADS-B:** Connect directly to the OpenSky network to track real flights live in the sky above major global airports (JFK, LHR, DEL, HND, SYD, etc.).

### 2. Controlling Playback
When watching Synthetic or High-Speed Playback scenarios, use the **Feed Speed** slider to slow down or speed up time. 
* *Note: The speed slider is disabled during Live ADS-B mode, as live feeds update exactly as fast as the real-world APIs allow.*

### 3. Recording Your Own Snapshots
While watching a **Live ADS-B** feed, the server will constantly maintain a rolling 20-minute buffer of everything that happens (this is your DVR Buffer). 
* Click the **Record New Snapshot** button to instruct the server to save the next 20 minutes of real-world data to a permanent file on your hard drive. 
* Once it finishes, refresh your browser, and you will see your new recording appear under the **High-Speed Playback** section for you to watch whenever you want!

## Understanding the Radar Scope
The map uses specific symbology to help you understand what the engine is thinking:

* **Dark Green Dots:** Raw radar pings. These are unfiltered detections (or sometimes ground clutter).
* **Grey/Blue Rings:** Tentative tracks. The engine has spotted something but is waiting for more consecutive pings to verify it isn't a bird or a glitch.
* **Cyan Boxes:** Confirmed tracks. The aircraft is fully verified. You will see its recent history path, its predicted forward trajectory, and a label showing its Callsign and Altitude (in km).
* **Yellow Boxes (CST):** Coasting tracks. The radar temporarily lost sight of the aircraft, so the engine is predicting where it went using physics until it regains visual contact.
