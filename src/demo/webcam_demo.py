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


def fingers_up(landmarks, handedness=None):
    """
    Return booleans for each finger. Accepts optional `handedness` ('Left'/'Right')
    to improve thumb detection. Backwards compatible with older callers.
    """
    # Delegate to the handedness-aware implementation (defined below)
    try:
        return fingers_up_with_handedness(landmarks, handedness)
    except NameError:
        # Fallback simple implementation if the detailed helper isn't available
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        up = [False] * 5
        for i, (t, p) in enumerate(zip(tips, pips)):
            try:
                if landmarks[t][1] < landmarks[p][1]:
                    up[i] = True
            except Exception:
                up[i] = False
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
            # Request up to 2 hands so we can obtain handedness for left/right controls
            options = HandLandmarkerOptions(base_options=base_options,
                                            num_hands=2,
                                            running_mode=RunningMode.VIDEO)
            task_hand_landmarker = HandLandmarker.create_from_options(options)
            mp_draw = None
            print("Using MediaPipe Tasks HandLandmarker with model:", model_path)
        except Exception as e:
            print("Failed to initialize MediaPipe Tasks HandLandmarker:", e)
            return
    else:
        mp_hands = mp_solutions.hands
        # allow detection of two hands so handedness info is available
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
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
    show_handedness_debug = False
    gesture_buffer = deque(maxlen=cfg["M_smoothing"])  # last M classified gestures
    last_major = "none"
    last_point = None
    color_idx = 0
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
    draw_color = colors[color_idx]
    color_names = ["green", "red", "blue", "yellow"]

    # hold timers
    gesture_start_time = {}
    recording = False
    session = {"session_id": int(time.time()), "frames": []}
    live_label = "none"

    # Frame-confirmation buffers and debounce for two-hand control
    N_confirm = int(cfg.get("N_activate", 5))
    debounce_s = float(cfg.get("debounce_s", 0.35))
    left_buffer = deque(maxlen=N_confirm)
    right_buffer = deque(maxlen=N_confirm)
    left_confirmed = "LOCK"
    right_confirmed = "none"
    last_mode_change_ts = 0.0
    # If True, prefer position (x coordinate) to decide left/right when handedness labels
    # are missing or appear inconsistent. This helps when the frame is mirrored.
    force_position_handedness = bool(cfg.get("force_position_handedness", True))

    print("Press 'q' to quit, 'c' toggle debug, 'r' start/stop recording, keys 1-5 to label while recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Two-hand mapping: produce `left_hand` and `right_hand` landmarks (pixel coords)
        gesture = "none"
        landmarks_px = None
        left_hand = None
        right_hand = None
        res = None

        if use_tasks_local:
            try:
                from mediapipe.tasks.python import vision as mp_vision
                if 'TensorImage' in globals() and TensorImage is not None:
                    timg = TensorImage.create_from_array(rgb)
                    detection = task_hand_landmarker.detect_for_video(timg, int(time.time() * 1000))
                else:
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
                handedness = getattr(detection, "handedness", None) or getattr(detection, "handedness_list", None) or getattr(detection, "handednesses", None)
                if multi_hand_landmarks:
                    for i, hand in enumerate(multi_hand_landmarks):
                        try:
                            lm_iter = getattr(hand, "landmark", hand)
                        except Exception:
                            lm_iter = hand
                        lm_list = []
                        for lm in lm_iter:
                            x = getattr(lm, "x", None)
                            y = getattr(lm, "y", None)
                            z = getattr(lm, "z", 0.0)
                            if x is None or y is None:
                                continue
                            lm_list.append((int(x * W), int(y * H), z))

                        # determine label if available
                        label = None
                        try:
                            if handedness and i < len(handedness):
                                h = handedness[i]
                                label = getattr(h, "category_name", None) or getattr(h, "label", None)
                                # some structures include classification list
                                if label is None:
                                    cls = getattr(h, "classification", None)
                                    if cls and len(cls) > 0:
                                        label = getattr(cls[0], "label", None)
                        except Exception:
                            label = None

                        target = None
                        # decide by position (x mean) and handedness label if available
                        mean_x = None
                        try:
                            xs = [p[0] for p in lm_list]
                            mean_x = sum(xs) / len(xs)
                        except Exception:
                            mean_x = None
                        side_by_pos = None
                        if mean_x is not None:
                            side_by_pos = "left" if mean_x < (W / 2) else "right"

                        if label:
                            lab = str(label).lower()
                            label_side = "right" if lab.startswith("r") else "left"
                            # If configured to prefer position-based handedness and there's a mismatch,
                            # use the position side because flipping/mirroring can make labels unreliable.
                            if force_position_handedness and side_by_pos is not None and label_side != side_by_pos:
                                target = side_by_pos
                            else:
                                target = label_side
                        else:
                            # fallback: use visual side
                            if side_by_pos is not None:
                                target = side_by_pos
                            else:
                                # fallback ordering
                                if right_hand is None:
                                    target = "right"
                                else:
                                    target = "left"

                        if target == "right":
                            right_hand = [(float(x), float(y), float(z)) for (x, y, z) in lm_list]
                        else:
                            left_hand = [(float(x), float(y), float(z)) for (x, y, z) in lm_list]
                        if show_handedness_debug:
                            print(f"TASK hand[{i}] mean_x={mean_x} label={label} -> target={target}")
            except Exception:
                left_hand = None
                right_hand = None
        else:
            # Legacy solutions API path
            res = hands.process(rgb)
            mhl = getattr(res, "multi_hand_landmarks", None)
            mhd = getattr(res, "multi_handedness", None)
            if mhl:
                for i, hand_landmarks in enumerate(mhl):
                    lm_list = landmarks_to_pixels(hand_landmarks.landmark, W, H)
                    label = None
                    try:
                        if mhd and i < len(mhd):
                            c = mhd[i].classification[0]
                            label = getattr(c, "label", None) or getattr(c, "category_name", None)
                    except Exception:
                        label = None

                    # decide by position (x mean) and handedness label if available
                    mean_x = None
                    try:
                        xs = [p[0] for p in lm_list]
                        mean_x = sum(xs) / len(xs)
                    except Exception:
                        mean_x = None
                    side_by_pos = None
                    if mean_x is not None:
                        side_by_pos = "left" if mean_x < (W / 2) else "right"

                    if label:
                        lab = str(label).lower()
                        label_side = "right" if lab.startswith("r") else "left"
                        if force_position_handedness and side_by_pos is not None and label_side != side_by_pos:
                            target = side_by_pos
                        else:
                            target = label_side
                    else:
                        if side_by_pos is not None:
                            target = side_by_pos
                        else:
                            target = "right" if right_hand is None else "left"

                    if target == "right":
                        right_hand = [(float(x), float(y), float(z)) for (x, y, z) in lm_list]
                    else:
                        left_hand = [(float(x), float(y), float(z)) for (x, y, z) in lm_list]
                    if show_handedness_debug:
                        print(f"SOL hand[{i}] mean_x={mean_x} label={label} -> target={target}")

        # For actions we only use the RIGHT hand landmarks (do not fallback to left)
        action_landmarks = right_hand
        landmarks_px = action_landmarks

        # Frame-confirmation and smoothing
        now = time.time()

        # LEFT candidate (control): OPEN PALM -> UNLOCK, FIST -> LOCK
        if left_hand:
            lf = fingers_up(left_hand, "Left")
            # consider only four fingers for open/closed checks (ignore thumb)
            four_open = lf["index"] and lf["middle"] and lf["ring"] and lf["pinky"]
            four_any = lf["index"] or lf["middle"] or lf["ring"] or lf["pinky"]
            if four_open:
                left_cand = "UNLOCK"
            elif not four_any:
                left_cand = "LOCK"
            else:
                left_cand = left_confirmed
        else:
            left_cand = "LOCK"

        left_buffer.append(left_cand)
        if len(left_buffer) == left_buffer.maxlen and all(x == left_cand for x in left_buffer) and (now - last_mode_change_ts) > debounce_s:
            if left_confirmed != left_cand:
                left_confirmed = left_cand
                last_mode_change_ts = now

        # RIGHT candidate (actions) only when unlocked
        if right_hand and left_confirmed == "UNLOCK":
            rf = fingers_up(right_hand, "Right")
            # ignore thumb when checking for fist/erase
            four_any_r = rf["index"] or rf["middle"] or rf["ring"] or rf["pinky"]
            # New mapping:
            # - Index only -> draw
            # - Index + middle -> erase
            # - All four fingers down (fist) -> change color
            if not four_any_r:
                right_cand = "color"
            elif rf["index"] and rf["middle"] and not rf["ring"]:
                right_cand = "erase"
            elif rf["index"] and not rf["middle"]:
                right_cand = "draw"
            elif rf["thumb"] and not (rf["index"] or rf["middle"] or rf["ring"] or rf["pinky"]):
                right_cand = "clear"
            else:
                right_cand = "none"
        else:
            right_cand = "none"

        right_buffer.append(right_cand)
        if len(right_buffer) == right_buffer.maxlen and all(x == right_cand for x in right_buffer) and (now - last_mode_change_ts) > debounce_s:
            if right_confirmed != right_cand:
                right_confirmed = right_cand
                last_mode_change_ts = now

        # Determine active major gesture: only when unlocked
        if left_confirmed == "UNLOCK":
            major = right_confirmed
        else:
            major = "none"

        # smoothing over last M frames to reduce flicker
        gesture_buffer.append(major)
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

        # DRAW (use RIGHT hand only and only when LEFT is UNLOCK and RIGHT confirmed DRAW)
        if left_confirmed == "UNLOCK" and right_confirmed == "draw":
            if action_landmarks:
                try:
                    ix_f, iy_f, _ = action_landmarks[8]
                    ix, iy = int(ix_f), int(iy_f)
                    if last_point is None:
                        last_point = (ix, iy)
                    # ensure integer tuples for cv2
                    cv2.line(canvas, (int(last_point[0]), int(last_point[1])), (ix, iy), draw_color, int(cfg["draw_radius"]))
                    last_point = (ix, iy)
                except Exception:
                    last_point = None
        else:
            last_point = None

        # ERASE (continuous) - only when unlocked and right_confirmed==erase
        if left_confirmed == "UNLOCK" and right_confirmed == "erase":
            if action_landmarks:
                try:
                    ex_f, ey_f, _ = action_landmarks[8]
                    ex, ey = int(ex_f), int(ey_f)
                    cv2.circle(canvas, (ex, ey), int(cfg["erase_radius"]), (0, 0, 0), -1)
                except Exception:
                    pass

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

        # LOCK/UNLOCK UI indicator and confirmed mode
        try:
            box_w, box_h = 220, 60
            pad = 12
            bx0, by0 = pad, pad
            bx1, by1 = bx0 + box_w, by0 + box_h
            # background box
            cv2.rectangle(disp, (bx0, by0), (bx1, by1), (40, 40, 40), -1)
            status_color = (0, 0, 255) if left_confirmed == "LOCK" else (0, 200, 0)
            status_text = left_confirmed
            cv2.putText(disp, status_text, (bx0 + 10, by0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
            # confirmed right mode
            mode_text = right_confirmed.upper() if right_confirmed and right_confirmed != "none" else "-"
            cv2.putText(disp, f"Mode: {mode_text}", (bx0 + 120, by0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        except Exception:
            pass

        # Always show current color swatch and selected color name in the top-right corner
        try:
            sw_w, sw_h = 90, 50
            pad = 10
            x0 = W - sw_w - pad
            y0 = pad
            x1 = W - pad
            y1 = pad + sw_h
            # filled rectangle swatch
            cv2.rectangle(disp, (x0, y0), (x1, y1), draw_color, -1)
            # border
            cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 255, 255), 1)
            # color name text left of swatch
            cname = color_names[color_idx] if color_idx < len(color_names) else str(color_idx)
            cv2.putText(disp, f"Color: {cname}", (x0 - 200, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception:
            pass

        # Draw a larger, prominent gesture label at the top-left corner
        try:
            cv2.putText(disp, f"{major}".upper(), (10, H - (H - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        except Exception:
            # fallback won't block
            pass

        # draw landmarks for debug (legacy solutions only)
        if (not use_tasks_local) and (res is not None) and getattr(res, "multi_hand_landmarks", None) and show_debug:
            mp_draw.draw_landmarks(disp, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.imshow("AirDraw - demo", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            show_debug = not show_debug
        elif key == ord("h"):
            show_handedness_debug = not show_handedness_debug
            print("Handedness debug:", show_handedness_debug)
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
