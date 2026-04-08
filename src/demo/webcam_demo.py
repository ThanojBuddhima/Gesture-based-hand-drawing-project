#!/usr/bin/env python3
"""
Real-time webcam demo for gesture-based air drawing.

Requirements:
  pip install opencv-python mediapipe numpy pyyaml

Usage:
  python src/demo/webcam_demo.py

Controls:
  - 'q' : quit
  - 'c' : toggle show debug overlay
  - 'r' : start/stop recording (saves JSON session in data/raw/)
  - number keys 1-5 : set live-label while recording

Gesture mapping (rule-based):
  - Index up -> Draw
  - Index + Middle up -> Selection
  - Fist (no fingers) -> Erase (continuous)
  - Thumb up -> Clear (hold)
  - Three fingers up (index+middle+ring) -> Change color (on release)

This file implements smoothing, debounce, hold-duration checks, and a simple canvas.
"""

import os
import time
import json
from collections import deque, Counter

import cv2
import numpy as np
import yaml
import importlib
import urllib.request
import shutil


# mediapipe package layout can differ by installation. Try multiple import locations
# and surface a helpful error if none succeed.
mp_solutions = None
try:
    mp_module = importlib.import_module("mediapipe")
    mp_solutions = getattr(mp_module, "solutions", None)
except Exception:
    mp_module = None

if mp_solutions is None:
    candidates = [
        "mediapipe.python.solutions",
        "mediapipe.solutions",
        "google.mediapipe.python.solutions",
        "google.mediapipe.solutions",
    ]
    for mod in candidates:
        try:
            mp_solutions = importlib.import_module(mod)
            break
        except Exception:
            mp_solutions = None
if mp_solutions is None:
    # Try the new Tasks API (mediapipe.tasks)
    mp_tasks = None
    try:
        mp_tasks = getattr(mp_module, "tasks", None)
    except Exception:
        mp_tasks = None

    if mp_tasks is None:
        candidates = [
            "mediapipe.tasks",
            "mediapipe.python.tasks",
            "google.mediapipe.tasks",
            "google.mediapipe.python.tasks",
        ]
        for mod in candidates:
            try:
                mp_tasks = importlib.import_module(mod)
                break
            except Exception:
                mp_tasks = None

    if mp_tasks is None:
        raise ImportError(
            "Could not import MediaPipe 'solutions' or 'tasks' modules.\n"
            "Possible fixes:\n"
            " - Ensure you installed MediaPipe in the active environment: `pip install mediapipe`\n"
            " - Try a compatible version that exposes 'solutions' (or use the Tasks API): `pip install mediapipe`\n"
            "Run `python -c \"import mediapipe as mp; print(dir(mp))\"` to inspect installed package."
        )
    else:
        # We'll use Tasks API path later in main()
        mp_tasks_module = mp_tasks
        mp_solutions = None
        mp_solutions = None
        mp_tasks = mp_tasks_module
        # mark that tasks API is available
        USE_TASKS = True
else:
    USE_TASKS = False


def load_config(path=None):
    default = {
        "N_activate": 5,
        "N_deactivate": 3,
        "M_smoothing": 7,
        "erase_radius": 30,
        "draw_radius": 4,
        "T_clear_hold": 1.0,
        "T_erase_hold": 0.6,
    }
    if path and os.path.exists(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        default.update(cfg or {})
    return default


def landmarks_to_pixels(landmarks, w, h):
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h), lm.z))
    return pts


def fingers_up(landmarks):
    # landmarks: list of 21 (x,y,z) normalized
    # tips = 4,8,12,16,20; pip = 3,6,10,14,18
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    up = [False] * 5
    for i, (t, p) in enumerate(zip(tips, pips)):
        # For fingers except thumb, tip.y < pip.y means finger is up (image coords origin top-left)
        if landmarks[t][1] < landmarks[p][1]:
            up[i] = True
    # For thumb, orientation varies; above simple test is OK in many cases
    return {
        "thumb": up[0],
        "index": up[1],
        "middle": up[2],
        "ring": up[3],
        "pinky": up[4],
    }


def majority_vote(buffer):
    if len(buffer) == 0:
        return "none"
    cnt = Counter(buffer)
    return cnt.most_common(1)[0][0]


def save_session(session, out_dir="data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    path = os.path.join(out_dir, f"session_{ts}.json")
    with open(path, "w") as f:
        json.dump(session, f, indent=2)
    print(f"Saved session to {path}")


def main():
    cfg = load_config("configs/default.yaml")

    # Initialize hand detector depending on available MediaPipe API
    hands = None
    mp_draw = None
    use_tasks_local = globals().get("USE_TASKS", False)
    task_hand_landmarker = None

    if use_tasks_local:
        # Try to import Tasks vision API
        try:
            # import BaseOptions from tasks.python and HandLandmarker classes from vision
            from mediapipe.tasks.python import BaseOptions
            try:
                # prefer direct vision imports
                from mediapipe.tasks.python.vision import (
                    HandLandmarker,
                    HandLandmarkerOptions,
                    RunningMode,
                    TensorImage,
                )
            except Exception:
                # fallback to vision namespace
                from mediapipe.tasks.python import vision as mp_vision
                HandLandmarker = mp_vision.HandLandmarker
                HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
                RunningMode = mp_vision.RunningMode
                TensorImage = getattr(mp_vision, "TensorImage", None)
            # ensure model exists or try to download it
            def try_download_model(dest_path):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                candidate_urls = [
                    # Common hosting locations (may or may not exist for your platform)
                    "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
                    "https://storage.googleapis.com/mediapipe/hand_landmarker.task",
                ]
                for url in candidate_urls:
                    try:
                        print(f"Attempting to download model from {url} ...")
                        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib/3"})
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            with open(dest_path + ".tmp", "wb") as out_f:
                                shutil.copyfileobj(resp, out_f)
                        os.replace(dest_path + ".tmp", dest_path)
                        print("Downloaded model to", dest_path)
                        return True
                    except Exception as e:
                        print(f"Download from {url} failed: {e}")
                # if candidates failed, prompt user for a URL
                print("Automatic downloads failed. If you have a Hand Landmarker task model URL, paste it now; or press Enter to skip:")
                user_url = input().strip()
                if user_url:
                    try:
                        req = urllib.request.Request(user_url, headers={"User-Agent": "python-urllib/3"})
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            with open(dest_path + ".tmp", "wb") as out_f:
                                shutil.copyfileobj(resp, out_f)
                        os.replace(dest_path + ".tmp", dest_path)
                        print("Downloaded model to", dest_path)
                        return True
                    except Exception as e:
                        print("Download failed:", e)
                return False

            # Expect a local model file 'models/hand_landmarker.task'
            model_paths = [
                "models/hand_landmarker.task",
                "hand_landmarker.task",
            ]
            model_path = None
            for p in model_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            if model_path is None:
                print("MediaPipe Tasks API detected but no 'hand_landmarker.task' model found.")
                dest = "models/hand_landmarker.task"
                ok = try_download_model(dest)
                if not ok:
                    print("No model available. Exiting. You can manually download a model and place it at 'models/hand_landmarker.task'.")
                    return
                model_path = dest

            base_options = BaseOptions(model_asset_path=model_path)
            # Use VIDEO running mode for synchronous frame-by-frame detection
            options = HandLandmarkerOptions(base_options=base_options,
                                            num_hands=1,
                                            running_mode=RunningMode.VIDEO)
            task_hand_landmarker = HandLandmarker.create_from_options(options)
            mp_draw = None
            print("Using MediaPipe Tasks HandLandmarker with model:", model_path)
        except Exception as e:
            print("Failed to initialize MediaPipe Tasks HandLandmarker:", e)
            return
    else:
        mp_hands = mp_solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_draw = mp_solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    ret, frame = cap.read()
    if not ret:
        print("Empty frame")
        return

    H, W = frame.shape[:2]
    canvas = np.zeros_like(frame)

    show_debug = True
    gesture_buffer = deque(maxlen=cfg["M_smoothing"])  # last M classified gestures
    last_major = "none"
    last_point = None
    color_idx = 0
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
    draw_color = colors[color_idx]

    # hold timers
    gesture_start_time = {}
    recording = False
    session = {"session_id": int(time.time()), "frames": []}
    live_label = "none"

    print("Press 'q' to quit, 'c' toggle debug, 'r' start/stop recording, keys 1-5 to label while recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gesture = "none"
        landmarks_px = None

        if use_tasks_local:
            # Tasks API path
            try:
                from mediapipe.tasks.python import vision as mp_vision
                if 'TensorImage' in globals() and TensorImage is not None:
                    timg = TensorImage.create_from_array(rgb)
                    detection = task_hand_landmarker.detect_for_video(timg, int(time.time() * 1000))
                else:
                    # Try passing numpy array directly, or fallback to mediapipe.Image
                    try:
                        detection = task_hand_landmarker.detect_for_video(rgb, int(time.time() * 1000))
                    except Exception:
                        img_module = getattr(mp_module, 'Image', None)
                        if img_module is not None:
                            mp_img = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
                            detection = task_hand_landmarker.detect_for_video(mp_img, int(time.time() * 1000))
                        else:
                            raise
                multi_hand_landmarks = getattr(detection, "hand_landmarks", None)
                if multi_hand_landmarks and len(multi_hand_landmarks) > 0:
                    hand = multi_hand_landmarks[0]
                    # hand may have .landmark or be iterable
                    try:
                        lm_iter = getattr(hand, "landmark", hand)
                    except Exception:
                        lm_iter = hand
                    lm_list = []
                    for lm in lm_iter:
                        # some landmark objects have x,y,z attributes
                        x = getattr(lm, "x", None)
                        y = getattr(lm, "y", None)
                        z = getattr(lm, "z", 0.0)
                        if x is None or y is None:
                            continue
                        lm_list.append((x, y, z))
                    lm_list_px = [(int(x * W), int(y * H), z) for (x, y, z) in lm_list]
                    landmarks_px = [(float(x), float(y), float(z)) for (x, y, z) in lm_list_px]
                    fu = fingers_up(lm_list_px)
                else:
                    fu = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
            except Exception as e:
                fu = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
        else:
            # Legacy solutions API path
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                lm_list = landmarks_to_pixels(hand.landmark, W, H)
                landmarks_px = [(float(x), float(y), float(z)) for (x, y, z) in lm_list]

                fu = fingers_up(lm_list)

            # Rule-based classification with priority
            if not any(fu.values()):
                gesture = "erase"
            elif fu["index"] and fu["middle"] and not fu["ring"]:
                gesture = "selection"
            elif fu["index"] and not fu["middle"]:
                gesture = "draw"
            elif fu["thumb"] and not (fu["index"] or fu["middle"] or fu["ring"] or fu["pinky"]):
                gesture = "clear"
            elif fu["index"] and fu["middle"] and fu["ring"]:
                gesture = "color"
            else:
                gesture = "none"

        # smoothing / majority
        gesture_buffer.append(gesture)
        major = majority_vote(gesture_buffer)

        # recording: append frames only if configured to save all, or if there's
        # a live label set, or if smoothed gesture is not 'none'
        save_only = cfg.get("save_only_labeled_frames", True)
        if recording:
            should_save = True
            if save_only:
                should_save = (live_label != "none") or (major != "none")
            if should_save:
                session["frames"].append({
                    "timestamp": int(time.time() * 1000),
                    "gesture_raw": gesture,
                    "gesture_smooth": major,
                    "landmarks": landmarks_px,
                    "live_label": live_label,
                })

        # transitions and hold logic
        now = time.time()
        if major != last_major:
            gesture_start_time[major] = now

        # DRAW
        if major == "draw":
            if landmarks_px:
                ix, iy, _ = landmarks_px[8]
                if last_point is None:
                    last_point = (ix, iy)
                cv2.line(canvas, last_point, (ix, iy), draw_color, cfg["draw_radius"])
                last_point = (ix, iy)
        else:
            last_point = None

        # ERASE (continuous)
        if major == "erase":
            if landmarks_px:
                ex, ey, _ = landmarks_px[8]
                cv2.circle(canvas, (int(ex), int(ey)), cfg["erase_radius"], (0, 0, 0), -1)

        # COLOR change: on release (edge from color -> not color)
        if last_major == "color" and major != "color":
            # switch color
            color_idx = (color_idx + 1) % len(colors)
            draw_color = colors[color_idx]

        # CLEAR: require hold
        if major == "clear":
            start_t = gesture_start_time.get("clear", now)
            if now - start_t >= cfg["T_clear_hold"]:
                canvas = np.zeros_like(frame)
                gesture_start_time["clear"] = now + 9999  # avoid repeating

        # Compose display
        overlay = img.copy()
        # blend canvas
        mask = canvas.astype(bool)
        overlay[mask] = canvas[mask]

        disp = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

        if show_debug:
            cv2.putText(disp, f"Gesture: {major}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(disp, f"LiveLabel: {live_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(disp, f"Recording: {recording}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # draw landmarks for debug (legacy solutions only)
        if (not use_tasks_local) and (res is not None) and getattr(res, "multi_hand_landmarks", None) and show_debug:
            mp_draw.draw_landmarks(disp, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.imshow("AirDraw - demo", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            show_debug = not show_debug
        elif key == ord("r"):
            recording = not recording
            if not recording:
                # save session
                save_session(session)
                session = {"session_id": int(time.time()), "frames": []}
                live_label = "none"
            else:
                print("Recording started")
        elif key in [ord(str(i)) for i in range(1, 6)]:
            if recording:
                live_label = {"1": "draw", "2": "selection", "3": "erase", "4": "clear", "5": "color"}[chr(key)]
                print("Live label set to", live_label)

        last_major = major

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
