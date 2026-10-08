import io

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from panomoche.sharpness import DEFAULT_MAX_BLURRED_AREA, is_not_sharp, measure

from .conftest import textured
from .test_service import INFO, fake_score


def blurred(w=1024, h=768, sigma=2.5):
    return Image.fromarray(cv2.GaussianBlur(textured(w, h), (0, 0), sigma))


def detail_ratio(img, is_pano, mask_boxes=()):
    return measure(img, is_pano, mask_boxes)["sharpness"]


def test_blur_lowers_the_detail_ratio():
    sharp = detail_ratio(Image.fromarray(textured(1024, 768)), False)
    assert detail_ratio(blurred(sigma=1.0), False) < sharp
    assert detail_ratio(blurred(sigma=2.5), False) < detail_ratio(blurred(sigma=1.0), False)
    assert measure(blurred(sigma=2.5), False)["blurred_area"] >= 0.5
    assert measure(Image.fromarray(textured(1024, 768)), False)["blurred_area"] < 0.1


def test_partly_blurred_picture_is_judged_on_its_blurred_area():
    """Like Sony #303: a sharp strip in the middle must not hide a blurred foreground."""
    a = textured(1024, 768)
    a[384:] = cv2.GaussianBlur(a[384:], (0, 0), 3)  # bottom half blurred
    a[:192] = cv2.GaussianBlur(a[:192], (0, 0), 3)  # and the top quarter
    m = measure(Image.fromarray(a), False)
    assert m["blurred_area"] >= 0.5 and is_not_sharp(m["blurred_area"], 0.5)


def test_privacy_boxes_are_ignored():
    a = textured(1024, 768)
    a[:, :512] = cv2.GaussianBlur(a[:, :512], (0, 0), 6)  # left half blurred like a huge SGBlur patch
    img = Image.fromarray(a)
    assert detail_ratio(img, False, [[0, 0, 0.5, 1]]) > detail_ratio(img, False)


def test_threshold_rules():
    assert is_not_sharp(0.6, 0.5) and is_not_sharp(0.5, 0.5) and not is_not_sharp(0.4, 0.5)
    assert not is_not_sharp(0.9, 1) and not is_not_sharp(None, 0.5)


def test_blurry_upload_is_excluded_without_scoring():
    from panomoche.service import create_app

    calls = []

    def score(prep):
        calls.append(1)
        return fake_score(prep)

    client = TestClient(create_app(score, INFO))
    buf = io.BytesIO()
    blurred().save(buf, format="JPEG", quality=90)
    body = client.post("/predict", files={"picture": ("b.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert body["excluded"] == "not_sharp" and body["blurred_area"] >= 0.5
    assert client.get("/api/info").json()["max_blurred_area"] == DEFAULT_MAX_BLURRED_AREA


def test_cli_predict_excludes_blurry_pictures(tmp_path, monkeypatch):
    import panomoche.scoring
    from panomoche.cli import main
    from panomoche.panoramax import read_jsonl

    monkeypatch.setattr(panomoche.scoring, "make_scorer", lambda *a, **kw: (fake_score, INFO))
    pics = tmp_path / "pics"
    pics.mkdir()
    blurred().save(pics / "blurry.jpg", quality=90)
    Image.fromarray(textured(1024, 768)).save(pics / "sharp.jpg", quality=90)
    out = tmp_path / "p.jsonl"
    main(["predict", str(pics), "--out", str(out), "--no-sequence"])
    rows = {r["id"]: r for r in read_jsonl(out)}
    assert rows["blurry"]["excluded"] == "not_sharp" and not rows["sharp"]["excluded"]
    main(["predict", str(pics), "--out", str(out), "--no-sequence", "--max-blurred-area", "1"])
    assert not any(r.get("excluded") for r in read_jsonl(out))


def test_enlarged_small_picture_is_measured_at_its_original_size():
    """Panoramax's sd asset is 2048 px wide: a sharp 512 px original arrives enlarged fourfold."""
    photo = cv2.GaussianBlur(textured(512, 384).astype(np.float32), (0, 0), 0.6)  # a camera's slight softness
    small = Image.fromarray(np.clip(photo, 0, 255).astype(np.uint8))
    enlarged = small.resize((2048, 1536), Image.Resampling.BICUBIC)
    native = measure(small, False)["blurred_area"]
    at_original_size = measure(enlarged, False, original_width=512)["blurred_area"]
    as_delivered = measure(enlarged, False)["blurred_area"]
    assert at_original_size <= native + 0.05  # as sharp as it ever was
    assert as_delivered >= 0.5  # measured as delivered, the enlargement would look blurred


def test_cpbd_drops_with_blur():
    yy, xx = np.mgrid[0:768, 0:1024]
    a = np.repeat((((xx // 24 + yy // 24) % 2) * 200 + 25).astype(np.uint8)[..., None], 3, -1)  # high-contrast edges
    sharp = measure(Image.fromarray(a), False)["cpbd"]
    soft = measure(Image.fromarray(cv2.GaussianBlur(a, (0, 0), 1.5)), False)["cpbd"]
    assert sharp > 0.9 and soft < 0.2


def test_no_edges_left_means_no_cpbd():
    assert measure(blurred(sigma=2.5), False)["cpbd"] is None  # the blurred area then decides alone


def test_cpbd_must_confirm_the_blurred_area():
    assert is_not_sharp(0.40, 0.33, cpbd=0.10)  # both agree
    assert not is_not_sharp(0.40, 0.33, cpbd=0.30)  # soft texture, but edges look sharp: kept
    assert is_not_sharp(0.25, 0.33, cpbd=0.05)  # smaller blurred area, clearly blurred edges
    assert not is_not_sharp(0.10, 0.33, cpbd=0.05)  # edges alone never exclude
    assert is_not_sharp(0.40, 0.33, cpbd=None)  # no edges measured: the blurred area decides
