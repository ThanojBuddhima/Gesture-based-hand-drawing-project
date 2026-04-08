"""Feature extraction for gesture dataset.

Reads JSON session files produced by `src/data/collect.py` (under `data/raw/`) and
produces per-session CSV files and a combined NumPy archive (`features.npz`) in
`data/processed/`.

Features per frame:
- 42 raw coordinates: flattened normalized x,y for 21 landmarks (x1,y1,...)
- 5 distances: wrist (0) to each fingertip (4,8,12,16,20)
- 5 angles: per finger angle (thumb,index,middle,ring,pinky) computed from triplets

Usage:
  python3 src/train/prepare_data.py --input data/raw --output data/processed --combine

"""

import argparse
import json
import math
import os
from pathlib import Path
import csv

import numpy as np


def load_session(path):
    with open(path, "r") as f:
        return json.load(f)


def to_np_landmarks(landmarks):
    # landmarks may be a list of dicts {x,y,z} or a list of [x,y,z] lists/tuples.
    out = []
    for lm in landmarks:
        if isinstance(lm, dict):
            x = lm.get("x", 0.0)
            y = lm.get("y", 0.0)
            z = lm.get("z", 0.0)
        else:
            # assume sequence-like
            try:
                x = lm[0]
                y = lm[1]
                z = lm[2] if len(lm) > 2 else 0.0
            except Exception:
                x = y = z = 0.0
        out.append([x, y, z])
    arr = np.array(out, dtype=float)
    return arr


def pair_distance(a, b):
    return float(np.linalg.norm(a - b))


def angle_between(a, b, c):
    # angle at point b between vectors ba and bc
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nb = np.linalg.norm(bc)
    if na == 0 or nb == 0:
        return 0.0
    cosang = np.dot(ba, bc) / (na * nb)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))


def extract_features_from_landmarks(landmarks):
    # landmarks: numpy array shape (21,3)
    lm = landmarks
    # Use normalized coordinates (x,y) as provided by collector if normalized, else pixel coords — both ok
    coords = lm[:, :2].flatten()  # 42 values

    # distances: wrist (0) to fingertips 4,8,12,16,20
    wrist = lm[0, :2]
    fingertips_idx = [4, 8, 12, 16, 20]
    dists = [pair_distance(wrist, lm[i, :2]) for i in fingertips_idx]

    # angles: use triplets per finger (proximal joints)
    # thumb: points 1-2-3 (angle at 2)
    # index: 5-6-7 (angle at 6)
    # middle: 9-10-11 (angle at 10)
    # ring: 13-14-15 (angle at 14)
    # pinky: 17-18-19 (angle at 18)
    angle_triplets = [(1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)]
    angles = [angle_between(lm[a, :2], lm[b, :2], lm[c, :2]) for (a, b, c) in angle_triplets]

    features = np.concatenate([coords, np.array(dists, dtype=float), np.array(angles, dtype=float)])
    return features


def process_all(input_dir, output_dir, combine=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_list = []
    y_list = []
    meta = []

    for f in sorted(input_dir.glob("session_*.json")):
        s = load_session(f)
        rows = []
        labels = []
        for fr in s.get("frames", []):
            lm = fr.get("landmarks")
            if not lm:
                continue
            arr = to_np_landmarks(lm)
            if arr.shape[0] != 21:
                # skip malformed frames
                continue
            feats = extract_features_from_landmarks(arr)
            rows.append(feats)
            labels.append(fr.get("label", "none"))

        if rows:
            rows_np = np.vstack(rows)
            labels_np = np.array(labels, dtype=object)
            # Write per-session CSV (features) and labels
            session_base = output_dir / f.stem
            feats_csv = session_base.with_suffix(".features.csv")
            labels_csv = session_base.with_suffix(".labels.csv")
            np.savetxt(feats_csv, rows_np, delimiter=",", fmt="%.6f")
            with open(labels_csv, "w", newline="") as lf:
                writer = csv.writer(lf)
                writer.writerow(["label"])  # header
                for lab in labels_np:
                    writer.writerow([lab])

            meta.append({"session": f.name, "n_frames": rows_np.shape[0], "features_file": str(feats_csv.name), "labels_file": str(labels_csv.name)})

            X_list.append(rows_np)
            y_list.append(labels_np)

    if combine and X_list:
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        np.savez(output_dir / "features.npz", X=X, y=y)
        print(f"Wrote combined features: {output_dir / 'features.npz'} (n_samples={X.shape[0]})")

    # write meta JSON
    try:
        import json as _json
        with open(output_dir / "meta.json", "w") as mf:
            _json.dump(meta, mf, indent=2)
    except Exception:
        pass

    print(f"Processed {len(meta)} sessions. Output directory: {output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw", help="Input sessions directory")
    p.add_argument("--output", default="data/processed", help="Output directory for processed features")
    p.add_argument("--combine", action="store_true", help="Combine all sessions into one features.npz")
    args = p.parse_args()
    process_all(args.input, args.output, combine=args.combine)


if __name__ == "__main__":
    main()
