# Processed data

This folder contains processed feature artifacts generated from `data/raw/` by
`src/train/prepare_data.py`.

Files produced

- `session_<ts>.features.csv` — per-frame numeric features (one row per frame).
- `session_<ts>.labels.csv` — per-frame labels corresponding to the features.
- `features.npz` — combined NumPy archive with arrays `X` (n_samples x n_features) and `y` (n_samples,).
- `meta.json` — metadata about processed sessions.

The feature vector layout:
- 0..41: flattened x,y coordinates for 21 landmarks (x1,y1,x2,y2,...)
- 42..46: distances from wrist (0) to fingertips (4,8,12,16,20)
- 47..51: angles (degrees) for thumb,index,middle,ring,pinky computed at proximal joints
