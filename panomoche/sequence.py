"""Per-sequence summary. Comparing a picture with its neighbours is part of the
single decision in ``decision.py``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

SEQUENCE_ISSUE = "sequence_outlier"


def sequence_summary(rows: list[dict], key: str = "quality_score") -> list[dict]:
    """Per-sequence median score and exclusion count, worst sequence first.

    A uniformly poor sequence (e.g. a dashcam filming through the windshield)
    has no outliers, so it only shows up here, through its low median.
    """
    by_seq: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("collection"):
            by_seq[r["collection"]].append(r)
    out = []
    for c, rs in by_seq.items():
        scores = [r[key] for r in rs if r.get(key) is not None]
        out.append(
            {
                "collection": c,
                "pictures": len(rs),
                "median": round(float(np.median(scores)), 4) if scores else None,
                "excluded": sum(bool(r.get("excluded")) for r in rs),
                "camera": rs[0].get("camera"),
            }
        )
    return sorted(out, key=lambda s: (s["median"] is None, s["median"] or 0.0))
