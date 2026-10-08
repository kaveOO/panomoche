"""How the merged waviness and advertisement checks feed the panomoche decision."""

import cv2
import numpy as np
import pytest
from PIL import Image

from panomoche.cli import main
from panomoche.decision import Rules, decide

from .test_decision import picture, sequence, status
from .test_service import review_client  # noqa: F401  (fixture)
from .test_waviness_tool import facade, warp


def wobbly_sequence(n=21, wavy_every=1):
    rows = sequence(n, "wobble", topiq=0.6, blurred=0.05)
    for i, r in enumerate(rows):
        r["waviness_label"] = "wavy" if i % wavy_every == 0 else "no_strong_evidence"
    return rows


def test_wavy_picture_in_a_wobbling_stretch_is_excluded():
    """Sony handlebar sequence: half of the pictures bend together."""
    rows = decide(wobbly_sequence(wavy_every=2))
    assert status(rows[10]) == "wavy" and status(rows[11]) == "kept"


def test_lone_wavy_picture_is_kept():
    """Wide-angle lens, tree trunks or a curved building: one picture, not the sequence."""
    rows = wobbly_sequence(wavy_every=100)  # only the first picture is wavy
    rows[10]["waviness_label"], rows[0]["waviness_label"] = "wavy", "no_strong_evidence"
    assert status(decide(rows)[10]) == "kept"
    assert status(decide([picture(1, None, waviness_label="wavy")])[0]) == "kept"  # judged alone


def test_wavy_needs_30_percent_of_the_neighbours():
    """A clean wide-angle Paris sequence had 4 wavy neighbours out of 20: kept."""
    rows = wobbly_sequence(wavy_every=100)
    for i in (10, 3, 6, 14, 18):  # the picture + 4 of its 20 neighbours
        rows[i]["waviness_label"] = "wavy"
    assert status(decide(rows)[10]) == "kept"
    for i in (1, 2):  # 6 of 20 neighbours
        rows[i]["waviness_label"] = "wavy"
    assert status(decide(rows)[10]) == "wavy"
    assert status(decide(rows, Rules(min_wavy_neighbours=2.0))[10]) == "kept"  # rule off


def test_360_pictures_are_not_checked_for_waviness():
    from panomoche.waviness import check

    img = Image.fromarray(cv2.cvtColor(warp(facade(), amount=16), cv2.COLOR_BGR2RGB))
    assert check(img, is_pano=False)["waviness_label"] == "wavy"
    assert check(img, is_pano=True) == {"waviness_label": None, "waviness_score": None}


def test_advertisement_is_excluded_whatever_the_context():
    rows = sequence(21, "s", topiq=0.6, blurred=0.05)
    rows[10]["ad_rejected"] = True
    rows[11]["ad_rejected"] = False
    decide(rows)
    assert status(rows[10]) == "advertisement" and status(rows[11]) == "kept"
    assert status(decide([picture(1, None, ad_rejected=True)])[0]) == "advertisement"


def test_unavailable_ad_model_passes(monkeypatch):
    from panomoche import advertising

    monkeypatch.setattr(advertising, "_ENGINE", {"engine": None, "error": "no model"})
    out = advertising.check(Image.new("RGB", (200, 200)))
    assert out["ad_rejected"] is False and out["ad_checked"] is False


@pytest.mark.parametrize("tool", ["waviness", "ads", "ads-setup", "waviness-ui", "ads-ui"])
def test_merged_tools_keep_their_command_lines(tool, capsys):
    with pytest.raises(SystemExit) as done:
        main([tool, "--help"])
    assert done.value.code == 0 and "usage: panomoche " + tool in capsys.readouterr().out


def test_waviness_command_writes_reports(tmp_path):
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "waviness" / "straight-street.jpg"
    with pytest.raises(SystemExit) as done:
        main(["waviness", str(example), "--output-dir", str(tmp_path)])
    assert done.value.code == 0 and (tmp_path / "001-straight-street.json").is_file()


def test_upload_runs_waviness_and_sequence_comparison_uses_it():
    import io

    from fastapi.testclient import TestClient

    from panomoche.service import create_app

    from .test_service import INFO, fake_score

    client = TestClient(create_app(fake_score, INFO, waviness=True))
    _, png = cv2.imencode(".png", warp(facade(), amount=16))
    body = client.post("/predict", files={"picture": ("wobble.png", io.BytesIO(png.tobytes()), "image/png")}).json()
    assert body["waviness_label"] == "wavy" and body["excluded"] is None  # alone: benefit of the doubt
    pics = [{**body, "id": f"u{i}"} for i in range(8)]  # a whole wobbling sequence
    seq = client.post("/api/sequence", json={"pictures": pics}).json()["pictures"]
    assert all(p["excluded"] == "wavy" for p in seq)
    assert client.get("/api/info").json()["waviness_check"] is True


def test_stray_quote_in_ocr_output_does_not_hide_the_following_words(monkeypatch):
    """Tesseract 5.3 reads decorative marks as '"'; CSV quoting then swallowed "BOOK NOW", "50% OFF"..."""
    import subprocess

    from panomoche.advertising import classifier

    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    words = ["SALE", '"', "BOOK", "NOW", "50%", "OFF", "Limited", "offer"]
    tsv = header + "".join(f"5\t1\t1\t1\t1\t{i}\t0\t0\t400\t60\t95\t{w}\n" for i, w in enumerate(words))
    monkeypatch.setattr(classifier.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(
        classifier.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout=tsv, stderr="")
    )
    ocr = classifier.ocr_evidence(Image.new("RGB", (800, 600)))
    assert ocr["words"] == len(words)
    assert {"50% off", "limited offer", "sale"} <= set(ocr["promotional_terms"])


def test_duplicates_and_known_pictures(tmp_path):
    from panomoche.dedupe import drop_duplicates, known_ids

    from .conftest import textured

    a = textured(320, 240, seed=1)
    b = np.ascontiguousarray(a[::-1, ::-1])  # different picture: rotated half a turn
    paths = {}
    for name, arr, q in (("a", a, 90), ("a_again", a, 90), ("a_recompressed", a, 70), ("b", b, 90)):
        paths[name] = tmp_path / f"{name}.jpg"
        Image.fromarray(arr).save(paths[name], quality=q)
    recs = [{"id": n, "path": str(p)} for n, p in paths.items()]
    kept, dups = drop_duplicates(recs[1:], reference=recs[:1])
    assert [r["id"] for r in kept] == ["b"]
    assert {d["id"]: d["duplicate_of"] for d in dups} == {"a_again": "a", "a_recompressed": "a"}
    (tmp_path / "preds.jsonl").write_text('{"id": "x"}\n{"id": "y"}\n')
    assert known_ids([tmp_path / "preds.jsonl", tmp_path / "missing.jsonl"]) == {"x", "y"}
