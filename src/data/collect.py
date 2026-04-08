"""Simple interactive landmark data collector.

Usage: Run `python src/data/collect.py` and use the on-screen keys to label/capture.

Controls (while running):
- Number keys `1`..`5`: set current label (see mapping below)
- `r`: toggle recording (when recording, frames with a non-'none' label are saved)
- `c`: capture one labeled frame immediately
- `q` or ESC: quit and save session

Output:
- A session file is written to `data/raw/session_<timestamp>.json` containing
  a JSON object with metadata and an array of frames: {timestamp, label, landmarks}

This script tries to support both MediaPipe `solutions` and the newer `tasks` API.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2


def ensure_data_dir():
    d = Path("data/raw")
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_mediapipe_detector():
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions"):
            hands = mp.solutions.hands.Hands(static_image_mode=False,
                                              max_num_hands=1,
                                              min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)

            def detect(image_rgb):
                results = hands.process(image_rgb)
                if not results.multi_hand_landmarks:
                    return None
                return results.multi_hand_landmarks[0]

            return detect
    except Exception:
        pass

    # Try Tasks API layout
    try:
        from mediapipe.tasks.python import vision

        model_path = Path("models/hand_landmarker.task")
        if not model_path.exists():
            print("WARNING: tasks HandLandmarker model not found at 'models/hand_landmarker.task'.")

        options = vision.HandLandmarkerOptions(model_asset_path=str(model_path), num_hands=1)
        detector = vision.HandLandmarker.create_from_options(options)

        def detect_tasks(image_rgb):
            try:
                from mediapipe.framework.formats import image as mp_image
                tensor_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=image_rgb)
                res = detector.detect(tensor_img)
            except Exception:
                res = detector.detect(image_rgb)

            if not res.hand_landmarks:
                return None
            return res.hand_landmarks[0]

        return detect_tasks
    except Exception:
        pass

    raise RuntimeError("No supported MediaPipe hand detector found. Install mediapipe or run the demo first to download the model.")


def landmarks_to_list(landmarks, image_w=None, image_h=None):
    out = []
    for lm in landmarks:
        x = getattr(lm, "x", None)
        y = getattr(lm, "y", None)
        z = getattr(lm, "z", None)
        if x is None:
            try:
                x, y, z = lm[0], lm[1], lm[2]
            except Exception:
                x = y = z = 0.0
        if image_w and image_h:
            out.append({"x": float(x * image_w), "y": float(y * image_h), "z": float(z)})
        else:
            out.append({"x": float(x), "y": float(y), "z": float(z)})
    return out


LABEL_MAP = {"1": "draw", "2": "select", "3": "erase", "4": "clear", "5": "change_color"}


def main():
    ensure_data_dir()
    try:
        detector = get_mediapipe_detector()
    except Exception as e:
        print(str(e))
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    current_label = "none"
    recording = False
    frames = []

    print("Controls: 1..5 set label, r toggle recording, c capture one frame, q quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            lm = detector(img_rgb)
        except Exception:
            lm = None

        if lm is not None:
            landmarks = landmarks_to_list(lm, image_w=w, image_h=h)
            for p in landmarks:
                cv2.circle(frame, (int(p["x"]), int(p["y"])), 2, (0, 255, 0), -1)
        else:
            landmarks = None

        status_text = f"Label:{current_label} | Recording:{recording} | Frames collected:{len(frames)}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        elif chr(key) in LABEL_MAP.keys():
            current_label = LABEL_MAP[chr(key)]
            print("Current label set to:", current_label)
        elif key == ord("r"):
            recording = not recording
            print("Recording:", recording)
        elif key == ord("c"):
            if landmarks is not None:
                frames.append({"timestamp": time.time(), "label": current_label, "landmarks": landmarks})
                print("Captured 1 frame for label:", current_label)
            else:
                print("No hand detected; frame not captured.")

        if recording and current_label != "none" and landmarks is not None:
            frames.append({"timestamp": time.time(), "label": current_label, "landmarks": landmarks})

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        session = {"created_at": datetime.utcnow().isoformat() + "Z", "n_frames": len(frames), "frames": frames}
        fname = ensure_data_dir() / f"session_{int(time.time())}.json"
        with open(fname, "w") as f:
            json.dump(session, f)
        print("Saved session:", str(fname))
    else:
        print("No frames captured; nothing saved.")


if __name__ == "__main__":
    main()
