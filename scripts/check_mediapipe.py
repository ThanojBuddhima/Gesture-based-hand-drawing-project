"""Small diagnostic to help debug mediapipe installation.

Run:
  python scripts/check_mediapipe.py

Paste the output when asking for help.
"""
import importlib
import sys

def try_import(name):
    try:
        m = importlib.import_module(name)
        return True, m
    except Exception as e:
        return False, str(e)

candidates = [
    "mediapipe",
    "mediapipe.solutions",
    "mediapipe.python.solutions",
    "google.mediapipe",
    "google.mediapipe.solutions",
]

for c in candidates:
    ok, out = try_import(c)
    print(f"Import {c}: {ok}")
    if ok:
        print("  module:", out)
        try:
            print("  attrs:", [a for a in dir(out) if not a.startswith("__")][:30])
        except Exception:
            pass
    else:
        print("  error:", out)

print("\nRun this to inspect mediapipe package object if 'mediapipe' imports:")
print("python -c \"import mediapipe as mp; print(type(mp), dir(mp)[:50])\"")
