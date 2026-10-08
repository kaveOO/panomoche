"""Advertisement check: rejects only obvious advertising graphics (SigLIP 2 + OCR); anything else passes.

``classifier.py`` is the merged standalone tool (its own CLI and ``server.py`` UI);
``check`` is what the panomoche decision uses.
"""

from __future__ import annotations

import threading

from PIL import Image

_ENGINE: dict = {}
_LOCK = threading.Lock()


def engine():
    """The SigLIP 2 engine, loaded once; None when the model or onnxruntime is missing (pictures then pass)."""
    with _LOCK:
        if "engine" not in _ENGINE:
            try:
                from .classifier import ContentFilter
                _ENGINE["engine"] = ContentFilter()
            except Exception as error:  # missing model data or package: the check is skipped, never a rejection
                _ENGINE["engine"], _ENGINE["error"] = None, str(error)
        return _ENGINE["engine"]


def check(img: Image.Image) -> dict:
    """``ad_rejected`` (only obvious advertising graphics), ``ad_score``, ``ad_checked`` and ``ad_reason``."""
    e = engine()
    if e is None:
        return {"ad_rejected": False, "ad_score": None, "ad_checked": False, "ad_reason": "check_unavailable"}
    result = e.analyze(img.convert("RGB"))
    return {"ad_rejected": result["ad_rejected"], "ad_score": result.get("ad_score"),
            "ad_checked": result["checked"], "ad_reason": result["reason_code"]}
