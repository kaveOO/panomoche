"""Waviness check: rolling-shutter wobble seen as shared bending of long vertical edges.

``detector.py`` is the merged standalone tool (its own CLI and ``server.py`` UI);
``check`` is what the panomoche decision uses.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .detector import VERSION, Settings, analyze, load_image  # noqa: F401


def check(img: Image.Image, is_pano: bool) -> dict:
    """``waviness_label`` (wavy / review / no_strong_evidence / insufficient_evidence) and ``waviness_score``.

    360° pictures are not checked: in an equirectangular picture, straight building edges are
    naturally curved, and on 1,300 Panoramax pictures every 360° "wavy" result was a clean picture.
    """
    if is_pano:
        return {"waviness_label": None, "waviness_score": None}
    result, *_ = analyze(cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR))
    return {"waviness_label": result["label"], "waviness_score": result["score"]}
