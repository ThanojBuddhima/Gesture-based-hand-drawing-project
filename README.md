# Gesture-Based Air Drawing — Balanced MVP

This repository hosts a real-time webcam air-drawing demo using MediaPipe hand landmarks and OpenCV. The project prioritizes a fast rule-based MVP for gesture control, then supports dataset collection and a simple ML pipeline for later improvements.

Repository layout

- `src/demo/webcam_demo.py` — main real-time demo and on-screen UI.
- `src/data/` — data collection utilities and preprocessing scripts.
- `src/train/` — training glue and feature extraction helpers.
- `docs/plan.me` — project plan and two-hand control specification.
- `models/` — (optional) MediaPipe Tasks model files (not committed by default).
- `data/raw/` and `data/processed/` — recorded sessions and processed features.

Quick start

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies (example):

```bash
pip install opencv-python mediapipe numpy pyyaml
```

3. Run the demo:

```bash
./venv/bin/python src/demo/webcam_demo.py
```

Runtime controls

- `q` — quit
- `c` — toggle debug overlay
- `h` — toggle handedness debug prints (terminal)
- `r` — start/stop recording (sessions saved to `data/raw/`)
# Gesture-Based Air Drawing

A lightweight, real-time webcam air-drawing demo that uses MediaPipe hand landmarks and OpenCV. The project focuses on a robust rule-based MVP for gesture control, with utilities for data collection and a simple training pipeline for future improvements.

Key features

- Two-hand control: left hand locks/unlocks the system; right hand performs drawing and editing actions only when unlocked.
- Rule-based gesture recognition with temporal smoothing and confirmation to reduce accidental activations.
- Live recording mode for dataset collection (per-frame landmark JSON output).
- Simple configurable UI with LOCK/UNLOCK status, color swatch, and fingertip markers.

Repository structure

- `src/demo/webcam_demo.py` — real-time demo, gesture logic and UI.
- `src/data/` — data collection and preprocessing utilities.
- `src/train/` — training helpers and feature preparation scripts.
- `configs/` — default runtime configuration (e.g., `default.yaml`).
- `docs/` — project documentation (planning and requirements).
- `models/` — (optional) local MediaPipe Task model files (not tracked by default).
- `data/raw/`, `data/processed/` — captured sessions and processed features.

Quick start (recommended)

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies (use `requirements.txt`):

```bash
pip install -r requirements.txt
```

3. Run the demo:

```bash
venv/bin/python src/demo/webcam_demo.py
```

Runtime keys

- `q` — quit
- `c` — toggle debug overlay
- `h` — toggle handedness debug prints (console)
- `r` — start/stop recording (saves JSON to `data/raw/`)
- `1`–`5` — set live label while recording

Gesture summary (high level)

Left hand (control)
- OPEN PALM → UNLOCK
- FIST → LOCK

Right hand (actions, only when UNLOCKED)
- INDEX → DRAW (index fingertip is cursor)
- INDEX + MIDDLE → ERASE
- FIST (index..pinky down) → CHANGE COLOR
- THUMBS-DOWN → CLEAR ALL (destructive — configurable hold/confirmation)

Notes on detection & configuration

- Fingers are detected using landmark geometry; thumb logic is handedness-aware.
- Temporal smoothing and confirmation are configured via `configs/default.yaml` (keys like `N_activate`, `debounce_s`, `M_smoothing`).
- If MediaPipe handedness labels appear inconsistent, `force_position_handedness` can prefer visual left/right positions.

Data collection & training

- Use `src/data/collect.py` to record labeled sessions into `data/raw/`.
- `src/train/prepare_data.py` prepares features (e.g., `data/processed/features.npz`) for training a lightweight classifier.

Development & contribution

- Work on feature branches off `test` and open pull requests to `main` when ready.
- Keep changes focused and provide a short local test plan in the PR description.

Troubleshooting

- Missing MediaPipe model: add `models/hand_landmarker.task` or install the appropriate `mediapipe` package.
- If gestures are unstable: tune `N_activate`, `M_smoothing`, and `debounce_s` in `configs/default.yaml`.

License & attribution

This repository is provided as-is for demonstration and research. Include an appropriate open-source license file if you intend to distribute or publish the project.

For details on gesture semantics and development notes, see `docs/plan.me` and `docs/requirements.md`.
