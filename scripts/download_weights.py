#!/usr/bin/env python
"""Pre-download DeepFace weights into DEEPFACE_HOME (./weights).

Run this once on deploy (as the serving user) so the first request doesn't pay
the download cost and the weights are readable by the process. Idempotent.

    python scripts/download_weights.py
"""
from __future__ import annotations

import os
import sys

# Make ./weights the DeepFace home BEFORE importing deepface.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings  # noqa: E402

settings = get_settings()
weights_dir = os.path.abspath(settings.deepface_home)
os.makedirs(weights_dir, exist_ok=True)
os.environ["DEEPFACE_HOME"] = weights_dir
print(f"DEEPFACE_HOME = {weights_dir}")

from deepface import DeepFace  # noqa: E402


def main() -> int:
    model = settings.face_model
    detector = settings.face_detector
    print(f"Building recognition model: {model} ...")
    DeepFace.build_model(model)

    # Trigger detector weight download via a tiny extract call.
    import numpy as np

    print(f"Priming detector backend: {detector} ...")
    try:
        DeepFace.extract_faces(
            img_path=np.zeros((160, 160, 3), dtype=np.uint8),
            detector_backend=detector,
            enforce_detection=False,
            align=True,
        )
    except Exception as exc:
        print(f"  (detector prime returned: {type(exc).__name__} — expected for a blank image)")

    print("Done. Weights are cached under", weights_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
