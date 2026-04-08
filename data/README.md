# Data collection

This folder stores raw captured landmark sessions used for training.

Guidelines

- Run `python src/data/collect.py` to open the interactive collector.
- Use keys `1`..`5` to set labels (draw/select/erase/clear/change_color).
- Press `r` to toggle continuous recording (only frames with a non-`none` label are saved while recording).
- Press `c` to capture one labeled frame immediately.
- Press `q` or ESC to quit; captured frames will be written to `data/raw/session_<timestamp>.json`.

Collect 100–200 clean samples per gesture across 1–3 users. Favor labeled, varied poses and lighting.
