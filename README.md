# Gesture-based-hand-drawing-project

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
- `1`-`5` — set live label while recording

Gesture mapping and two-hand control

Left (control) hand — locks/unlocks system
- OPEN PALM (index/middle/ring/pinky up) -> `UNLOCK`
- FIST (index..pinky down) -> `LOCK`
- If LEFT is not detected -> remain `LOCKED` (safety)

Right (action) hand — actions only when LEFT == `UNLOCK`
- INDEX (index finger up only) -> `DRAW` (track index tip)
- INDEX + MIDDLE -> `ERASE` (continuous)
- FIST (all four fingers down) -> `CHANGE COLOR` (cycles immediately)
- THUMBS-DOWN (right-hand thumb tip below IP joint) -> `CLEAR ALL` (immediate canvas wipe)
- THUMB UP -> `CLEAR` (if implemented as a hold action)

Notes on gesture detection

- Fingers (index/middle/ring/pinky) are considered "up" when the fingertip y is above the corresponding lower joint y (image origin is top-left).
- Thumb detection respects `handedness` to handle left/right thumb orientation.
- The demo applies smoothing and confirmation: a gesture must be present for `N_activate` frames and debounced by `debounce_s` seconds.

Configuration

If present, `configs/default.yaml` overrides defaults. Useful keys:

- `N_activate`: frames to confirm a gesture (default 5)
- `debounce_s`: minimal seconds between confirmed mode changes (default ~0.3)
- `M_smoothing`: majority vote window size
- `draw_radius`, `erase_radius`: sizes for drawing and erasing
- `T_clear_hold`: hold time for clear actions (seconds)
- `force_position_handedness`: prefer visual position (left/right) when MediaPipe labels are inconsistent

Data collection & training

- `src/data/collect.py` collects landmark frames with live labels (saved into `data/raw/`).
- `src/train/prepare_data.py` extracts features to `data/processed/features.npz`.
- Recommended ML baseline: Random Forest or small MLP trained on landmark features.

UI & Visual Feedback

- Prominent LOCK/UNLOCK status box and separate Mode box (single source of truth for current action).
- Color swatch at the top-right updates immediately when color changes.
- Pencil/eraser markers appear at the fingertip during draw/erase operations.
- Debug overlay (optional) shows `LiveLabel` and `Recording` for collection.

Troubleshooting

- If MediaPipe Tasks model is required but missing, add `models/hand_landmarker.task` or install a `mediapipe` package with Solutions API.
- If handedness appears swapped (mirrored camera), set `force_position_handedness: true` in `configs/default.yaml`.
- If thumbs-down clears unexpectedly, I can make the check normalized by hand bbox height or require a short hold.

Development notes

- Branch from `test` for features. Keep commits small and test locally before pushing.
- The project was developed and tested in a local venv. Adjust package versions as needed.
