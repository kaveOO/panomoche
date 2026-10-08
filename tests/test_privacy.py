"""SGBlur record reading, blurred share and the keep/exclude decision."""

import io
import json

import pytest
from PIL import Image

from panomoche.privacy import (
    NO_RECORD,
    blurred_share,
    is_excluded,
    parse_sgblur_header,
    parse_sgblur_record,
    privacy_from_file,
    privacy_from_url,
)
from panomoche.sequence import sequence_summary

from .conftest import textured


def jpeg_with_record(w: int, h: int, detections) -> bytes:
    """A JPEG ending like SGBlur leaves it: comment segment with str(info), then EOI."""
    buf = io.BytesIO()
    Image.fromarray(textured(w, h)).save(buf, format="JPEG")
    data = buf.getvalue()
    assert data.endswith(b"\xff\xd9")
    body = str(detections).encode()
    return data[:-2] + b"\xff\xfe" + (len(body) + 2).to_bytes(2, "big") + body + b"\xff\xd9"


REAL_308 = [{"class": "face", "confidence": 0.066, "xywh": [448, 0, 416, 384]}]  # as read from Panoramax


def test_reads_the_record_sgblur_appends():
    data = jpeg_with_record(2048, 1536, REAL_308)
    assert parse_sgblur_record(data) == REAL_308
    assert parse_sgblur_record(data[-500:]) == REAL_308  # the tail of the file is enough


def test_no_record_or_foreign_comment():
    buf = io.BytesIO()
    Image.fromarray(textured(64, 64)).save(buf, format="JPEG")
    assert parse_sgblur_record(buf.getvalue()) is None
    assert parse_sgblur_record(jpeg_with_record(64, 64, "made with my camera")) is None


def test_record_is_never_executed():
    data = jpeg_with_record(64, 64, "x")
    malicious = data.replace(b"'x'", b"__import__('os').system('echo pwned')")
    assert parse_sgblur_record(malicious) is None


def test_share_matches_the_real_308_picture():
    r = blurred_share(REAL_308, 2048, 1536, False)
    assert r["privacy_blur"] == round(416 * 384 / (2048 * 1536), 4) == 0.0508
    assert r["privacy_patches"] == 1 and r["privacy_record"] == "sgblur"
    assert r["privacy_boxes"] == [[0.2188, 0.0, 0.4219, 0.25]]


def test_only_blurred_classes_and_sizes_count():
    dets = [
        {"class": "face", "bbox": [0, 0, 100, 100]},
        {"class": "plate", "bbox": [50, 50, 150, 150]},  # overlaps the face: counted once
        {"class": "sign", "bbox": [500, 500, 900, 900]},  # signs are not blurred
        {"class": "face", "bbox": [300, 300, 310, 330]},  # under 12 px wide: SGBlur skips it
    ]
    r = blurred_share(dets, 1000, 1000, False)
    assert r["privacy_patches"] == 2
    assert r["privacy_blur"] == round((100 * 100 * 2 - 50 * 50) / 1e6, 4)


def test_360_share_is_over_the_whole_picture():
    """Like GoPro Fusion #227: a band-only share doubled the patches of 360° pictures."""
    dets = [{"class": "face", "bbox": [0, 0, 400, 300]}, {"class": "face", "bbox": [0, 900, 400, 1100]}]
    r = blurred_share(dets, 4000, 2000, True)
    assert r["privacy_blur"] == round((400 * 300 + 400 * 200) / (4000 * 2000), 4)


def test_missing_record_meaning_depends_on_the_file():
    buf = io.BytesIO()
    Image.fromarray(textured(64, 64)).save(buf, format="JPEG")
    assert privacy_from_file(buf.getvalue(), False) == NO_RECORD  # hd original: nothing found
    assert privacy_from_file(buf.getvalue(), False, original=False)["privacy_blur"] is None  # unknown


class FakeResponse:
    def __init__(self, content, status):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        pass

    def close(self):
        pass


class FakeSession:
    def __init__(self, data, ranges=True):
        self.data, self.ranges, self.calls = data, ranges, []

    def get(self, url, headers=None, **kw):
        self.calls.append(headers.get("Range"))
        if not self.ranges:
            return FakeResponse(self.data, 200)
        spec = headers["Range"].split("=")[1]
        if spec.startswith("-"):
            return FakeResponse(self.data[-int(spec[1:]) :], 206)
        a, b = (int(v) for v in spec.split("-"))
        return FakeResponse(self.data[a : b + 1], 206)


def test_remote_record_with_range_requests():
    data = jpeg_with_record(2048, 1536, REAL_308)
    s = FakeSession(data)
    assert privacy_from_url(s, "https://x/hd.jpg", False)["privacy_blur"] == 0.0508
    assert len(s.calls) == 2 and all(c.startswith("bytes=") for c in s.calls)


def test_server_without_ranges_needs_explicit_full_download():
    data = jpeg_with_record(2048, 1536, REAL_308)
    assert privacy_from_url(FakeSession(data, ranges=False), "u", False)["privacy_record"] == "no range support"
    assert (
        privacy_from_url(FakeSession(data, ranges=False), "u", False, allow_full_download=True)["privacy_blur"]
        == 0.0508
    )


def test_network_errors_mean_unknown():
    class Broken:
        def get(self, *a, **kw):
            raise ConnectionError("down")

    r = privacy_from_url(Broken(), "u", False)
    assert r["privacy_blur"] is None and r["privacy_record"].startswith("error")


def test_x_sgblur_header_format():
    header = json.dumps({"info": [{"class": "plate", "bbox": [0, 0, 20, 20]}], "model": {"name": "yolo11s"}})
    assert parse_sgblur_header(header) == [{"class": "plate", "bbox": [0, 0, 20, 20]}]
    assert parse_sgblur_header('[{"class": "face"}]') == [{"class": "face"}]


@pytest.mark.parametrize(
    "share,limit,excluded", [(0.12, 0.05, True), (0.05, 0.05, False), (None, 0.05, False), (0.5, None, False)]
)
def test_only_known_heavy_blur_excludes(share, limit, excluded):
    assert is_excluded({"privacy_blur": share}, limit) is excluded


def test_summary_of_fully_excluded_sequence():
    rows = [{"id": "a", "collection": "c", "excluded": "privacy_blur", "quality_score": None, "low_quality": False}]
    assert sequence_summary(rows) == [{"collection": "c", "pictures": 1, "median": None, "excluded": 1, "camera": None}]


def test_size_found_behind_large_metadata():
    """Some cameras put more than 64 KB of metadata before the JPEG frame header."""
    data = jpeg_with_record(800, 600, REAL_308)
    padding = b"".join(b"\xff\xe2" + (60002).to_bytes(2, "big") + b"\x00" * 60000 for _ in range(3))  # 3 APP2 blocks
    data = data[:2] + padding + data[2:]  # right after SOI, before the frame header
    assert Image.open(io.BytesIO(data)).size == (800, 600)
    r = privacy_from_url(FakeSession(data), "u", False)
    assert r["privacy_record"] == "sgblur" and r["privacy_patches"] == 1
