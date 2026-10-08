"""The single decision that sorts every picture: excluded or kept.

All measurements are taken first (metadata, privacy blur, orientation,
sharpness, advertisement, waviness, TOPIQ), then ``decide`` applies one set of
rules. A picture is excluded, with one reason in ``excluded``, when any of
these applies:

1. **Whatever the context**
   * ``labelled_blurred`` / ``labelled_excluded``: a reviewer marked the picture
     Blurred on the review page (``data/user_labels.json``, ``apply_labels``).
     Reviewer labels only ever exclude: an "OK" label never keeps a picture
     the rules exclude;
   * ``low_resolution``: below 15 px per degree (``metadata.py``);
   * ``advertisement``: an obvious advertising graphic: SigLIP 2 layout match
     plus OCR-read promotional wording (``advertising/``). Anything ambiguous,
     or unchecked (no model, no OCR), passes;
   * ``privacy_blur``: blurred faces/plates cover more than 5% of the picture;
   * ``wrong_orientation``: content rotated, with at least 99% confidence; for
     a 360° picture, at least 2 of 4 rendered horizon views (camera mounted
     pitched or rolled 90°, see ``orientation.py``);
   * ``not_sharp``: two sharpness measures agree, at a viewing size (1024 px,
     see ``sharpness.py``): 33% or more of the textured area blurred with a
     CPBD (share of edges that look sharp) under 0.15, or 20% or more with a
     CPBD under 0.12. At that size a dashcam's windshield haze stays around 14%
     blurred while real motion blur reaches 35%; pixel-level softness at 100%
     zoom (phone processing, enlarged files) is not counted, and CPBD keeps
     soft but clean textures (foliage, dusk, small phone-app pictures) from
     counting as blur.
2. **With neighbours** (at least 2 pictures of the same sequence within 10
   positions):
   * ``wavy``: rolling-shutter wobble (``waviness/``) on a flat picture,
     confirmed by at least 30% of its neighbours being wavy too. A wobbling
     mount affects a long stretch of a sequence (half of a handlebar
     sequence); a wide-angle lens, a curved building or tree trunks bend
     single pictures' lines (on 1,300 pictures, every lone wavy result was a
     clean picture, and a clean wide-angle Paris sequence had 5 of 21). 360°
     pictures are not checked (their straight edges are curved by the
     projection);
   * ``sequence_outlier``: TOPIQ, which varies a lot from camera to camera,
     0.12 or more below the neighbours' median.
3. **TOPIQ floor**, ``low_quality``: below 0.32 for 360° pictures and for any
   picture with neighbours, and below 0.40 for a flat picture alone. The
   second is a fallback (in Panoramax every picture has its sequence) and
   gives the benefit of the doubt: a dull camera's clean pictures score
   around 0.42, as does the rest of their sequence. Both were set by looking
   at every picture they catch: just above them are dim dusk 360° pictures
   and plain country roads that look acceptable.

Every other picture is kept. ``issues`` lists the TOPIQ findings (what
``tag`` would write to Panoramax), ``low_quality`` is true for every excluded
picture.

These rules were checked against the reviewer's examples: a forest in sun
glare, a washed-out van, a blurry dusk picture and a motion-blurred handlebar
picture (to exclude), and a 360° picture and a dashcam sequence filming
through a hazy windshield (to keep). A sequence shot through a side window
cannot be told apart from that dashcam with these measurements (same
softness, same blur, and the phone's heading points along the road): it needs
a model that understands the content, trained on reviewed examples. A
"sideways blur" measure was tried and dropped: it reacted to horizontal lines
(dashboards, horizons) rather than to motion.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .iqa import LOW_QUALITY
from .metadata import metadata_exclusion
from .orientation import is_wrong
from .privacy import is_excluded
from .sequence import SEQUENCE_ISSUE
from .sharpness import is_not_sharp


@dataclass
class Rules:
    max_privacy_blur: float = 0.05
    min_wavy_neighbours: float = 0.30  # share of neighbours that must be wavy too (above 1: rule off)
    min_orientation_confidence: float = 0.99
    max_gps_accuracy: float = 0.0  # recorded, not used (reviewer decision)
    min_pixel_density: float = 15
    max_blurred_area: float = 0.33
    confirm_cpbd: float = 0.15
    soft_blurred_area: float = 0.20
    soft_cpbd: float = 0.12
    max_topiq_below_neighbours: float = 0.12
    topiq_floor_pano: float = 0.32
    topiq_floor_flat_alone: float = 0.40
    window: int = 10
    min_neighbours: int = 2


def _neighbours(rows: list[dict], window: int) -> dict[int, list[dict]]:
    """Index of each row -> rows of the same sequence within ``window`` ranks (itself excluded)."""
    by_seq: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("collection") and r.get("rank") is not None:
            by_seq[r["collection"]].append(i)
    out: dict[int, list[dict]] = {}
    for idx in by_seq.values():
        for i in idx:
            out[i] = [rows[j] for j in idx if j != i and abs(rows[j]["rank"] - rows[i]["rank"]) <= window]
    return out


def _median(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(vals)) if vals else None


def _confirmed_wavy(r: dict, neighbours: list[dict], min_share: float) -> bool:
    """A wavy picture whose sequence neighbours are wavy too, at least ``min_share`` of the measured ones."""
    if r.get("waviness_label") != "wavy" or min_share > 1:
        return False
    measured = [n for n in neighbours if n.get("waviness_label") is not None]
    return len(measured) >= 2 and sum(n["waviness_label"] == "wavy" for n in measured) / len(measured) >= min_share


LABEL_EXCLUSIONS = {"blurred": "labelled_blurred", "exclude": "labelled_excluded"}


def apply_labels(rows: list[dict], labels: dict[str, str]) -> list[dict]:
    """Record each reviewer label on its row and exclude what the reviewer marked, in place.

    Works on rows already decided (the rules' own verdict is kept in ``rules_excluded``, so clearing
    a label restores it) as well as before ``decide``, which also honours ``reviewer_label``.
    """
    for r in rows:
        if "rules_excluded" not in r:
            r["rules_excluded"] = r.get("excluded")
        label = labels.get(r["id"])
        r["reviewer_label"] = label
        r["excluded"] = LABEL_EXCLUSIONS.get(label) or r["rules_excluded"]
        r["low_quality"] = bool(r["excluded"])
    return rows


def hard_exclusion(r: dict, rules: Rules) -> str | None:
    if r.get("reviewer_label") in LABEL_EXCLUSIONS:
        return LABEL_EXCLUSIONS[r["reviewer_label"]]
    reason = metadata_exclusion(r, rules.max_gps_accuracy, rules.min_pixel_density)
    if reason:
        return reason
    if r.get("ad_rejected"):
        return "advertisement"
    if is_excluded(r, rules.max_privacy_blur):
        return "privacy_blur"
    if is_wrong(r, rules.min_orientation_confidence):
        return "wrong_orientation"
    if is_not_sharp(
        r.get("blurred_area"),
        rules.max_blurred_area,
        r.get("cpbd"),
        rules.confirm_cpbd,
        rules.soft_blurred_area,
        rules.soft_cpbd,
    ):
        return "not_sharp"
    return None


def decide(rows: list[dict], rules: Rules | None = None) -> list[dict]:
    """Set ``excluded`` (reason or None) / ``issues`` / ``low_quality`` / ``context`` on every row, in place."""
    rules = rules or Rules()
    near = _neighbours(rows, rules.window)
    for i, r in enumerate(rows):
        r.pop("excluded", None)
        r.pop("rules_excluded", None)
        r.pop("sequence", None)
        issues = [x for x in r.get("issues", []) if x not in (LOW_QUALITY, SEQUENCE_ISSUE)]
        nb = near.get(i, [])
        context = len(nb) >= rules.min_neighbours
        r["context"] = {"neighbours": len(nb)} if context else None
        reason = hard_exclusion(r, rules)
        score = r.get("quality_score")
        if not reason and context and _confirmed_wavy(r, nb, rules.min_wavy_neighbours):
            reason = "wavy"
        if not reason and context:
            topiq_med = _median(nb, "quality_score")
            r["context"].update(topiq_median=topiq_med, blurred_median=_median(nb, "blurred_area"))
            if score is not None and topiq_med is not None and topiq_med - score >= rules.max_topiq_below_neighbours:
                issues.append(SEQUENCE_ISSUE)
        if not reason and score is not None:
            floor = rules.topiq_floor_pano if (r.get("is_pano") or context) else rules.topiq_floor_flat_alone
            if score < floor:
                issues.append(LOW_QUALITY)
        if reason:
            issues = []
        r["excluded"] = reason or (issues[0] if issues else None)
        r["issues"] = issues
        r["low_quality"] = bool(r["excluded"])
    return rows
