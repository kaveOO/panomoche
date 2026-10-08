"""Review page for predictions: static file (``predict --html``) or served by ``serve``.

Both use the same template (``web/review.html``): filters per issue, sequence
order to see an excluded picture among its neighbours, a per-sequence table, and
a full-size viewer with keyboard navigation.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

from .sequence import sequence_summary

VIEWER_URL = "https://api.panoramax.xyz/#focus=pic&pic={id}"
CARD_FIELDS = (
    "id",
    "collection",
    "rank",
    "camera",
    "is_pano",
    "issues",
    "low_quality",
    "quality_score",
    "scores",
    "sequence",
    "excluded",
    "privacy_blur",
    "privacy_patches",
    "privacy_record",
    "sharpness",
    "blurred_area",
    "cpbd",
    "waviness_label",
    "waviness_score",
    "ad_rejected",
    "ad_score",
)

ImageUrl = Callable[[dict, str], "str | None"]  # (prediction, "thumb" | "full") -> URL


def template(name: str) -> str:
    return files("panomoche").joinpath("web", name).read_text(encoding="utf-8")


def script_json(data) -> str:
    """JSON safe to embed in a <script> element."""
    return json.dumps(data, separators=(",", ":")).replace("</", "<\\/")


def render_review(
    predictions: list[dict],
    title: str,
    image_url: ImageUrl,
    upload_url: str | None = None,
) -> str:
    pictures = []
    for p in predictions:
        card = {k: p.get(k) for k in CARD_FIELDS}
        card.update(
            thumb=image_url(p, "thumb"),
            full=image_url(p, "full"),
            viewer_url=VIEWER_URL.format(id=p["id"]) if p.get("collection") else None,
        )
        pictures.append(card)
    data = {
        "title": title,
        "pictures": pictures,
        "sequences": sequence_summary(predictions),
        "upload_url": upload_url,
    }
    return (
        template("review.html").replace("__TITLE__", title.replace("<", "&lt;")).replace("__DATA__", script_json(data))
    )


def html_report(predictions: list[dict], out: Path, title: str = "Panoramax picture quality") -> None:
    """Standalone review page.

    Local pictures are linked relative to ``out``; Panoramax thumbnails are used when known.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    def image_url(p: dict, size: str) -> str | None:
        local = os.path.relpath(p["path"], out.parent) if p.get("path") else None
        return (p.get("thumb_url") or local) if size == "thumb" else (local or p.get("thumb_url"))

    out.write_text(render_review(predictions, title, image_url), encoding="utf-8")
