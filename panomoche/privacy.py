"""How much of a picture Panoramax's anonymisation blurred, read from SGBlur's own record.

Panoramax blurs faces and licence plates with SGBlur before publishing a
picture. SGBlur appends what it detected to the JPEG as a comment segment
(``FF FE``) at the very end of the file: a Python-literal list such as

    [{'class': 'face', 'confidence': 0.066, 'xywh': [448, 0, 416, 384]}]

Faces and plates at least 12 px wide and high are blurred; road signs are only
detected. The record survives in the picture's ``hd`` asset (the original
file) but not in the re-encoded ``sd``/``thumb`` derivatives, and the API
doesn't expose it. It sits at the end of the file, so an HTTP range request
for the last 64 KB is enough on instances that support ranges (OSM-FR does,
IGN didn't when checked in Oct 2026).

On upload, a Panoramax backend gets the same detections from SGBlur's
``x-sgblur`` response header and can pass them to ``/predict`` directly.

Recognising the patches from pixels alone was tried first and doesn't work:
most cameras deliver pictures as smooth as a blurred patch, so a pixel
detector found only 14 of 40 large synthetic patches.
"""

from __future__ import annotations

import ast
import io
import json

from PIL import Image

BLURRED_CLASSES = {"face", "plate"}
MIN_SIDE = 12  # SGBlur leaves smaller detections unblurred
RANGE_BYTES = 65536  # a JPEG comment segment is at most 64 KB
DEFAULT_MAX_PRIVACY_BLUR = 0.05


def parse_sgblur_record(data: bytes) -> list[dict] | None:
    """Detections from the last comment segment of a JPEG (or of its tail), None if there is none."""
    end = data.rfind(b"\xff\xd9")
    i = data.rfind(b"\xff\xfe", 0, end if end > 0 else len(data))
    while i >= 0:
        size = int.from_bytes(data[i + 2 : i + 4], "big")
        text = data[i + 4 : i + 2 + size].decode("utf-8", "replace").strip()
        try:
            value = ast.literal_eval(text)  # literals only: safe on untrusted input
        except ValueError, SyntaxError:
            value = None
        if isinstance(value, list) and all(isinstance(d, dict) and "class" in d for d in value):
            return value
        i = data.rfind(b"\xff\xfe", 0, i)
    return None


def parse_sgblur_header(text: str) -> list[dict]:
    """Detections from SGBlur's ``x-sgblur`` header (JSON with an ``info`` list) or a bare JSON list."""
    value = json.loads(text)
    return value.get("info", []) if isinstance(value, dict) else value


def _box(d: dict) -> tuple[float, float, float, float] | None:
    if "bbox" in d and len(d["bbox"]) == 4:
        x0, y0, x1, y1 = d["bbox"]
    elif "xywh" in d and len(d["xywh"]) == 4:
        x, y, w, h = d["xywh"]
        x0, y0, x1, y1 = x, y, x + w, y + h
    else:
        return None
    if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
        return None
    return float(x0), float(y0), float(x1), float(y1)


def _union_area(boxes: list[tuple[float, float, float, float]]) -> float:
    """Area covered by axis-aligned boxes, overlaps counted once."""
    xs = sorted({v for b in boxes for v in (b[0], b[2])})
    area = 0.0
    for xa, xb in zip(xs, xs[1:]):
        covered, top, bottom = 0.0, None, None
        for y0, y1 in sorted((b[1], b[3]) for b in boxes if b[0] <= xa and b[2] >= xb):
            if top is None or y0 > bottom:
                covered += 0.0 if top is None else bottom - top
                top, bottom = y0, y1
            else:
                bottom = max(bottom, y1)
        covered += 0.0 if top is None else bottom - top
        area += covered * (xb - xa)
    return area


def blurred_share(detections: list[dict], width: int, height: int, is_pano: bool | None = None) -> dict:
    """Share of the whole picture hidden by blurred faces and plates.

    For 360° pictures too: measuring over the horizon band only (half the
    picture) counted the same patches twice and excluded clean 360° pictures
    (GoPro Fusion #227: 5.0% of the band, 2.7% of the picture). Box coordinates
    are in the stored JPEG orientation, so ``width``/``height`` must be too.
    ``is_pano`` is accepted for compatibility and not used.
    """
    boxes = [b for d in detections if d.get("class") in BLURRED_CLASSES and (b := _box(d))]
    clipped = [(max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)) for x0, y0, x1, y1 in boxes]
    return {
        "privacy_blur": round(_union_area(clipped) / (width * height), 4),
        "privacy_patches": len(boxes),
        "privacy_record": "sgblur",
        # Relative to the whole picture. Not a privacy leak: the blurred patches are visible anyway.
        "privacy_boxes": [
            [round(v, 4) for v in (x0 / width, y0 / height, x1 / width, y1 / height)] for x0, y0, x1, y1 in boxes
        ],
    }


NO_RECORD = {"privacy_blur": 0.0, "privacy_patches": 0, "privacy_record": "none", "privacy_boxes": []}  # nothing found


def unknown(reason: str) -> dict:
    return {"privacy_blur": None, "privacy_patches": None, "privacy_record": reason, "privacy_boxes": []}


def privacy_from_file(data: bytes, is_pano: bool | None, original: bool = True) -> dict:
    """Blurred share of a complete JPEG file.

    ``original``: the file is a Panoramax ``hd`` asset, so a missing record
    means SGBlur found nothing. For any other file (re-encoded ``sd``, a
    picture from elsewhere) a missing record means the share is unknown.
    """
    detections = parse_sgblur_record(data[-RANGE_BYTES:])
    if detections is None:
        return dict(NO_RECORD) if original else unknown("no SGBlur record in file")
    width, height = Image.open(io.BytesIO(data)).size
    return blurred_share(detections, width, height, is_pano)


def _remote_size(session, url: str) -> tuple[int, int]:
    """Picture size from the start of the file. Metadata (EXIF thumbnail, maker notes, XMP) can
    push the JPEG frame header past the first 64 KB, so the read grows until it is found."""
    for size in (RANGE_BYTES, 8 * RANGE_BYTES, 32 * RANGE_BYTES):
        head = session.get(url, headers={"Range": f"bytes=0-{size - 1}"}, timeout=60)
        head.raise_for_status()
        try:
            return Image.open(io.BytesIO(head.content)).size
        except OSError:
            continue
    raise OSError("JPEG frame header not found in the first 2 MB")


def privacy_from_url(session, url: str, is_pano: bool | None, allow_full_download: bool = False) -> dict:
    """Read SGBlur's record from a remote ``hd`` asset with two small range requests.

    A server that ignores ranges would send the whole file; that is only
    accepted with ``allow_full_download`` (several MB per 360° picture).
    """
    try:
        tail = session.get(url, headers={"Range": f"bytes=-{RANGE_BYTES}"}, timeout=60, stream=True)
        tail.raise_for_status()
        if tail.status_code != 206:
            if not allow_full_download:
                tail.close()
                return unknown("no range support")
            return privacy_from_file(tail.content, is_pano)
        detections = parse_sgblur_record(tail.content)
        if detections is None:
            return dict(NO_RECORD)
        width, height = _remote_size(session, url)
        return blurred_share(detections, width, height, is_pano)
    except Exception as exc:  # network error, truncated header...
        return unknown(f"error: {type(exc).__name__}")


def is_excluded(privacy: dict | None, max_privacy_blur: float | None) -> bool:
    """Only a known heavy blur excludes a picture; an unknown share keeps it."""
    share = (privacy or {}).get("privacy_blur")
    return max_privacy_blur is not None and share is not None and share > max_privacy_blur
