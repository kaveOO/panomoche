"""Web app: upload page, /predict, /api/sequence, review page and images (fake scorer, no model download)."""

import io
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from panomoche.panoramax import write_jsonl
from panomoche.report import render_review
from panomoche.service import create_app

from .conftest import textured

INFO = {
    "scorer": "topiq",
    "model": "panomoche-topiq_nr-spaq/test",
    "issues": ["low_quality", "sequence_outlier"],
    "thresholds": {"low_quality": 0.35},
}


def fake_score(prep):
    """Brightness as quality: dark pictures are 'low quality'. Same contract as TopiqPredictor."""
    q = round(float(prep.rgb.mean()) / 255, 4)
    views = [round(float(v.mean()) / 255, 4) for v in prep.views()]
    issues = ["low_quality"] if q < 0.35 else []
    return {
        "issues": issues,
        "low_quality": bool(issues),
        "quality_score": q,
        "scores": {"topiq": q},
        "view_scores": views,
        "model": INFO["model"],
    }


def jpeg(w=640, h=480, dark=False) -> bytes:
    arr = textured(w, h)
    if dark:
        arr = arr // 6
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(create_app(fake_score, INFO))


@pytest.fixture
def review_client(tmp_path):
    rows = []
    for i in range(8):
        path = tmp_path / f"p{i}.jpg"
        path.write_bytes(jpeg(800, 600))
        rows.append(
            {
                "id": f"pic-{i}",
                "collection": "seq-1",
                "rank": i + 1,
                "camera": "Test cam",
                "is_pano": False,
                "path": str(path),
                "issues": ["sequence_outlier"] if i == 3 else [],
                "low_quality": i == 3,
                "excluded": "sequence_outlier" if i == 3 else None,
                "quality_score": 0.3 if i == 3 else 0.6,
                "scores": {"topiq": 0.6},
                "sequence": {"median": 0.6, "z": -5.0 if i == 3 else 0.1, "neighbours": 7},
                "model": INFO["model"],
            }
        )
    write_jsonl(tmp_path / "preds.jsonl", rows)
    return TestClient(create_app(fake_score, INFO, tmp_path / "preds.jsonl"))


def embedded(html: str, element_id: str) -> dict:
    m = re.search(rf'<script id="{element_id}" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1))


def test_upload_page_has_model_info(client):
    r = client.get("/")
    assert r.status_code == 200 and "__INFO__" not in r.text
    assert embedded(r.text, "info-data")["model"] == INFO["model"]


def test_predict_returns_prediction_and_tags(client):
    r = client.post("/predict", files={"picture": ("a.jpg", jpeg(dark=True), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200 and body["low_quality"] and body["issues"] == ["low_quality"]
    assert body["width"] == 640 and body["is_pano"] is False and body["elapsed_ms"] >= 0
    assert body["semantics"][0] == {"key": "quality_issue", "value": "low_quality", "action": "add"}


def test_predict_detects_and_forces_projection(client):
    pano = client.post("/predict", files={"picture": ("p.jpg", jpeg(2048, 1024), "image/jpeg")}).json()
    assert pano["is_pano"] is True and len(pano["view_scores"]) == 4
    forced = client.post(
        "/predict", data={"is_pano": "false"}, files={"picture": ("p.jpg", jpeg(2048, 1024), "image/jpeg")}
    ).json()
    assert forced["is_pano"] is False and len(forced["view_scores"]) == 1


def test_predict_rejects_unreadable_file(client):
    r = client.post("/predict", files={"picture": ("a.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422


def test_sequence_endpoint_decides_uploads_as_one_sequence(client):
    pics = [
        {"id": f"u{i}", "quality_score": 0.6, "blurred_area": 0.05, "is_pano": False, "issues": []} for i in range(10)
    ]
    pics[4].update(quality_score=0.3, issues=["sequence_outlier"])  # stale flag from a previous run
    pics[7]["issues"] = ["sequence_outlier"]  # stale flag must disappear
    pics[2]["blurred_area"] = 0.40  # 35 points above its neighbours
    body = client.post("/api/sequence", json={"pictures": pics}).json()
    by = {p["id"]: p for p in body["pictures"]}
    assert "sequence_outlier" in by["u4"]["issues"] and by["u7"]["issues"] == []
    assert by["u2"]["excluded"] == "not_sharp"
    assert body["summary"]["pictures"] == 10 and body["params"]["drop"] == 0.12


def test_review_needs_predictions(client):
    assert client.get("/review").status_code == 404


def test_review_page_and_images(review_client):
    r = review_client.get("/review")
    assert r.status_code == 200
    data = embedded(r.text, "data")
    assert len(data["pictures"]) == 8 and data["sequences"][0]["excluded"] == 1
    pic = data["pictures"][3]
    assert pic["thumb"] == "images/pic-3?size=thumb" and pic["viewer_url"].endswith("pic=pic-3")

    thumb = review_client.get("/images/pic-3?size=thumb")
    assert thumb.status_code == 200 and Image.open(io.BytesIO(thumb.content)).width == 480
    full = review_client.get("/images/pic-3?size=full")
    assert Image.open(io.BytesIO(full.content)).width == 800


def test_images_only_serves_listed_pictures(review_client):
    assert review_client.get("/images/nope").status_code == 404
    assert review_client.get("/images/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_review_data_cannot_break_out_of_script():
    rows = [
        {
            "id": "x</script><script>alert(1)</script>",
            "issues": [],
            "low_quality": False,
            "quality_score": 0.5,
            "scores": {},
            "path": "x.jpg",
        }
    ]
    html = render_review(rows, "t", lambda r, s: "x.jpg")
    assert "</script><script>alert(1)" not in html
    assert embedded(html, "data")["pictures"][0]["id"] == rows[0]["id"]


# --- privacy blur -------------------------------------------------------------------------------

from .test_privacy import jpeg_with_record  # noqa: E402

HEAVY = [{"class": "face", "confidence": 0.07, "xywh": [0, 0, 300, 300]}]  # 300²/(640·480) ≈ 29 %
LIGHT = [{"class": "plate", "confidence": 0.8, "xywh": [100, 100, 40, 20]}]


def counting_client():
    calls = []

    def score(prep):
        calls.append(1)
        return fake_score(prep)

    from panomoche.decision import Rules

    return TestClient(create_app(score, INFO, rules=Rules(max_privacy_blur=0.05))), calls


def test_heavily_blurred_upload_is_excluded():
    client, _ = counting_client()
    body = client.post("/predict", files={"picture": ("a.jpg", jpeg_with_record(640, 480, HEAVY), "image/jpeg")}).json()
    assert body["excluded"] == "privacy_blur" and body["semantics"] == [] and body["issues"] == []
    assert body["privacy_blur"] > 0.05 and body["privacy_boxes"]
    assert body["quality_score"] is not None  # every picture is measured, excluded or not


def test_backend_can_pass_sgblur_output():
    client, _ = counting_client()
    header = json.dumps({"info": [{"class": "face", "bbox": [0, 0, 300, 300]}]})
    body = client.post("/predict", data={"sgblur": header}, files={"picture": ("a.jpg", jpeg(), "image/jpeg")}).json()
    assert body["excluded"] == "privacy_blur"
    bad = client.post("/predict", data={"sgblur": "not json"}, files={"picture": ("a.jpg", jpeg(), "image/jpeg")})
    assert bad.status_code == 422


def test_lightly_blurred_and_unknown_uploads_are_scored():
    client, calls = counting_client()
    light = client.post(
        "/predict", files={"picture": ("a.jpg", jpeg_with_record(640, 480, LIGHT), "image/jpeg")}
    ).json()
    assert not light["excluded"] and light["privacy_record"] == "sgblur" and light["privacy_blur"] < 0.05
    plain = client.post("/predict", files={"picture": ("b.jpg", jpeg(), "image/jpeg")}).json()
    assert plain["privacy_blur"] is None and plain["privacy_record"] == "no SGBlur record in file"
    assert len(calls) == 2
    assert client.get("/api/info").json()["max_privacy_blur"] == 0.05


def test_cli_predict_excludes_heavily_blurred_pictures(tmp_path, monkeypatch):
    import panomoche.scoring
    from panomoche.cli import main
    from panomoche.panoramax import read_jsonl

    calls = []

    def fake_make_scorer(*a, **kw):
        def score(prep):
            calls.append(1)
            return fake_score(prep)

        return score, INFO

    monkeypatch.setattr(panomoche.scoring, "make_scorer", fake_make_scorer)
    pics = tmp_path / "pics"
    pics.mkdir()
    (pics / "heavy.jpg").write_bytes(jpeg_with_record(640, 480, HEAVY))
    (pics / "light.jpg").write_bytes(jpeg_with_record(640, 480, LIGHT))
    (pics / "plain.jpg").write_bytes(jpeg())
    out = tmp_path / "preds.jsonl"
    main(["predict", str(pics), "--out", str(out), "--no-sequence"])
    rows = {r["id"]: r for r in read_jsonl(out)}
    assert rows["heavy"]["excluded"] == "privacy_blur" and rows["heavy"]["quality_score"] is not None
    assert not rows["light"]["excluded"] and not rows["plain"]["excluded"]
    assert rows["heavy"]["quality_score"] is not None  # measured, then excluded by the decision

    main(["predict", str(pics), "--out", str(out), "--no-sequence", "--max-privacy-blur", "0.5"])
    assert not any(r.get("excluded") for r in read_jsonl(out))


def test_no_manual_labelling(review_client):
    """Verdicts come from the rules only: no labelling endpoint, no Blurred / OK buttons."""
    assert review_client.post("/api/labels", json={"id": "pic-1", "label": "blurred"}).status_code in (404, 405)
    page = review_client.get("/review").text
    assert "labels_url" not in embedded(page, "data") and "Blurred" not in page


def _served(tmp_path, rows, **kw):
    for r in rows:
        r.setdefault("path", str(tmp_path / f"{r['id']}.jpg"))
        Path(r["path"]).write_bytes(jpeg(800, 600))
    write_jsonl(tmp_path / "preds.jsonl", rows)
    return TestClient(create_app(fake_score, INFO, tmp_path / "preds.jsonl", **kw))


def test_review_page_recomputes_stored_verdicts_with_the_servers_rules(tmp_path):
    """Verdicts saved by an older predict must not differ from what the upload page decides now."""
    rows = [
        {
            "id": "stale-excluded",
            "is_pano": False,
            "quality_score": 0.6,
            "blurred_area": 0.05,
            "cpbd": 0.4,
            "issues": [],
            "excluded": "not_sharp",
        },
        {
            "id": "stale-kept",
            "is_pano": False,
            "quality_score": 0.6,
            "blurred_area": 0.6,
            "cpbd": 0.05,
            "issues": [],
            "excluded": None,
        },
        {
            "id": "fetched",
            "is_pano": False,
            "quality_score": 0.6,
            "blurred_area": 0.05,
            "cpbd": 0.4,
            "issues": ["sequence_outlier"],
            "excluded": "sequence_outlier",
            "context_fetched": True,
        },
    ]
    pics = {p["id"]: p for p in embedded(_served(tmp_path, rows).get("/review").text, "data")["pictures"]}
    assert pics["stale-excluded"]["excluded"] is None and pics["stale-kept"]["excluded"] == "not_sharp"
    assert pics["fetched"]["excluded"] == "sequence_outlier"  # judged with neighbours not in the file: kept as is


def test_uploading_a_review_picture_gives_the_review_verdict(tmp_path):
    """Same picture, same information: metadata, privacy record and sequence neighbours come from the review data."""
    rows = [
        {
            "id": f"pic-{i}",
            "collection": "seq-1",
            "rank": i + 1,
            "is_pano": False,
            "quality_score": 0.6,
            "blurred_area": 0.05,
            "cpbd": 0.4,
            "issues": [],
            "original_size": [800, 600],
            "privacy_record": "sgblur",
            "privacy_blur": 0.0,
            "privacy_patches": 0,
            "privacy_boxes": [],
        }
        for i in range(8)
    ]
    client = _served(tmp_path, rows)
    dark = client.post("/predict", files={"picture": ("pic-3.jpg", jpeg(800, 600, dark=True), "image/jpeg")}).json()
    assert dark["collection"] == "seq-1" and dark["review_id"] == "pic-3" and dark["context"]["neighbours"] == 7
    assert dark["privacy_record"] == "sgblur" and dark["excluded"] == "sequence_outlier"  # vs its neighbours
    alone = client.post("/predict", files={"picture": ("other.jpg", jpeg(800, 600, dark=True), "image/jpeg")}).json()
    assert alone["context"] is None and alone["excluded"] == "low_quality"  # unknown picture: alone


def test_reviewer_labels_left_in_old_prediction_files_are_ignored(tmp_path):
    rows = [
        {
            "id": "pic-0",
            "is_pano": False,
            "quality_score": 0.6,
            "blurred_area": 0.05,
            "cpbd": 0.4,
            "issues": [],
            "excluded": "labelled_blurred",
            "rules_excluded": None,
            "reviewer_label": "blurred",
        }
    ]
    client = _served(tmp_path, rows)
    (pic,) = embedded(client.get("/review").text, "data")["pictures"]
    assert pic["excluded"] is None
    body = client.post("/predict", files={"picture": ("pic-0.jpg", jpeg(800, 600), "image/jpeg")}).json()
    assert body["excluded"] is None and "reviewer_label" not in body
