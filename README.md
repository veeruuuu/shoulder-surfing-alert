# Shoulder Surfing Alert

A webcam-based privacy tool that detects when someone behind or beside you is
looking at your screen and blurs it until they look away, plus hides the
screen if you step away from your desk.

## What it does

- Watches the webcam for faces that are **near**, **looking at the screen**,
  and **not you or someone you trust** — blurs the screen until they look
  away (about 0.7s to trigger, clears about 2s after the peeker leaves).
- Hides the screen if you step away from your desk for a few seconds
  (configurable; defaults to 4s).
- Dashboard to calibrate, enroll trusted people, review intruder snapshots,
  and tune settings live (no restart needed for most changes).
- Everything runs locally. No cloud, no accounts, no network calls except to
  your own machine's dashboard.

## Requirements

- Windows 10/11 (this build has only been tested on Windows; macOS code paths
  exist in `platform_utils.py` but are untested — see Known Limitations)
- Python 3.10 or 3.11
- A webcam
- About 500MB free disk for model files (downloaded once, not included in
  the repo)

## Setup (first time only)

```powershell
cd shoulder-surfing-alert
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts\download_models.py
```

The last command downloads two model files (MediaPipe face landmarker, SFace
recognizer) into `models/` — a few hundred MB, only needs to run once.

## Running it

```powershell
.venv\Scripts\Activate.ps1
python -m ssa.main
```

This starts the camera, the privacy overlay, and a local dashboard. The
terminal will print the dashboard URL (default `http://127.0.0.1:5000/`).
Open that in a browser.

**Important:** close any other app using your webcam (Camera app, Zoom,
Teams, browser tabs, etc.) before starting — see Known Limitations below.

Press `Ctrl+C` in the terminal to stop.

## First-time calibration

The app won't detect anything meaningful until you calibrate:

1. Open the dashboard, go to **Calibration**.
2. Sit alone in front of the camera at a normal distance (about 50cm).
3. Click **Calibrate**. This stores your face as the "owner" baseline.

Optionally, go to **Trusted Faces** to enroll anyone else who shouldn't
trigger alerts (same process: sit alone in frame, enter their name, enroll).

## Dashboard pages

- **Live** — camera preview, state badge (SAFE/ALERT/AWAY/PAUSED), pause/resume.
- **Calibration** — owner enrollment.
- **Trusted Faces** — enroll/remove people who never trigger alerts.
- **Intruder Log** — history of detected peeker events, with snapshots.
- **Settings & Stats** — raw config editor (hot-reloads within ~5s, no
  restart needed for most values) and a chart of events over time.

## Configuration

Defaults live in `config/defaults.yaml`. Runtime changes made via the
dashboard's Settings page are saved to `data/settings.json` and override the
defaults; most values (detection thresholds, delays, blur settings) apply
live within about 5 seconds. Camera resolution and model paths require a
restart.

## Known limitations

- **Camera sharing with other apps:** if another application (Camera app,
  video call software) has the webcam open at the same time, Windows may
  share access rather than refuse it — the app will show `camera: active`
  but detection may be starved (frames arrive at ~1fps, faces won't be
  detected, and the app may incorrectly show AWAY). Close other camera apps
  before running.
- **macOS:** not tested. Camera backend and screen-lock code paths exist for
  it in `platform_utils.py` but have not been verified.
- **Side-angle detection (about 45 degrees or more):** gaze detection is
  head-pose based, not true eye tracking — a face turned far enough away
  from the camera may not register as "looking" even if actually looking at
  the screen from an angle. See `docs/eval_results.md` scenario 4.
- **Settings hot-reload** does not cover camera resolution/index or model
  paths — those need an app restart.

## Running tests

```powershell
python -m pytest tests\ -v
```

## Project structure

See `SHOULDER_SURFING_ALERT.md` for the full design document, architecture,
and change log. `docs/eval_results.md` has evaluation methodology and (as
filled in) real test results.
