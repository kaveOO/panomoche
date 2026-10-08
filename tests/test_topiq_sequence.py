from PIL import Image

from panomoche.decision import decide
from panomoche.imaging import prepare
from panomoche.iqa import LOW_QUALITY, TopiqPredictor
from panomoche.panoramax import semantics_for
from panomoche.report import html_report
from panomoche.sequence import SEQUENCE_ISSUE

from .conftest import textured


def seq_rows(scores, collection="c1", ranks=None):
    ranks = ranks or range(1, len(scores) + 1)
    return [
        {
            "id": f"{collection}-{r}",
            "collection": collection,
            "rank": r,
            "issues": [],
            "low_quality": False,
            "quality_score": s,
        }
        for r, s in zip(ranks, scores)
    ]


def flagged(rows):
    return [r["id"] for r in rows if SEQUENCE_ISSUE in r["issues"]]


class FakeScorer:
    def __init__(self, mean, worst):
        self.mean, self.worst = mean, worst

    def score_picture(self, views):
        return {"views": len(views), "topiq_nr-spaq": self.mean, "topiq_nr-spaq_min": self.worst}


def test_topiq_predictor_uses_worst_view():
    prep = prepare(Image.fromarray(textured(2048, 1024)))
    ok = TopiqPredictor(0.35, scorer=FakeScorer(0.62, 0.51))(prep)
    assert ok["issues"] == [] and ok["quality_score"] == 0.51 and ok["scores"] == {"topiq": 0.51, "topiq_mean": 0.62}
    bad = TopiqPredictor(0.35, scorer=FakeScorer(0.40, 0.30))(prep)
    assert bad["issues"] == [LOW_QUALITY] and bad["low_quality"]
    assert bad["model"].startswith("panomoche-topiq_nr-spaq/")


def test_threshold_based_issues_get_no_confidence_tag():
    pred = {
        "issues": [LOW_QUALITY, SEQUENCE_ISSUE],
        "scores": {"topiq": 0.3, "topiq_mean": 0.4},
        "model": "panomoche-topiq_nr-spaq/0.1.0",
    }
    tags = semantics_for(pred)
    assert [t["key"] for t in tags] == [
        "quality_issue",
        "detection_model[quality_issue=low_quality]",
        "quality_issue",
        "detection_model[quality_issue=sequence_outlier]",
    ]


def test_static_report_links_pictures_and_sequences(tmp_path):
    rows = seq_rows([0.6] * 10)
    rows[4].update(issues=[SEQUENCE_ISSUE], low_quality=True, quality_score=0.3)
    for r in rows:
        r.update(path=str(tmp_path / f"{r['id']}.jpg"), scores={"topiq": r["quality_score"]})
    decide(rows)
    local_row = {"id": "x", "path": str(tmp_path / "x.jpg"), "issues": [], "low_quality": False, "quality_score": 0.6}
    out = tmp_path / "reports" / "r.html"
    html_report(rows + [local_row], out)
    import json
    import re

    data = json.loads(
        re.search(r'<script id="data" type="application/json">(.*?)</script>', out.read_text(), re.S).group(1)
    )
    pics = {p["id"]: p for p in data["pictures"]}
    assert SEQUENCE_ISSUE in pics["c1-5"]["issues"] and pics["c1-5"]["excluded"] == SEQUENCE_ISSUE
    assert pics["x"]["thumb"] == "../x.jpg"  # relative to the report, so the static file works offline
    assert pics["x"]["viewer_url"] is None and pics["c1-5"]["viewer_url"].endswith("pic=c1-5")
    assert data["sequences"][0]["collection"] == "c1" and data["sequences"][0]["excluded"] == 1


def test_sequence_summary_surfaces_uniformly_poor_sequences():
    from panomoche.sequence import sequence_summary

    dashcam = seq_rows([0.44] * 12, collection="dashcam")
    phone = seq_rows([0.75] * 12, collection="phone")
    phone[3]["excluded"] = "sequence_outlier"
    summary = sequence_summary(dashcam + phone + [{"id": "x", "quality_score": 0.1, "excluded": "low_quality"}])
    assert [(s["collection"], s["median"], s["excluded"]) for s in summary] == [
        ("dashcam", 0.44, 0),
        ("phone", 0.75, 1),
    ]


def test_flat_pictures_use_their_own_threshold():
    flat = prepare(Image.fromarray(textured(1024, 768)))
    pano = prepare(Image.fromarray(textured(2048, 1024)))
    score = FakeScorer(0.39, 0.38)
    assert TopiqPredictor(0.35, scorer=score)(flat)["issues"] == [LOW_QUALITY]  # 0.38 < 0.40 for a flat photo
    assert TopiqPredictor(0.35, scorer=score)(pano)["issues"] == []  # but fine for a 360° picture
    assert TopiqPredictor(0.35, scorer=score, abs_threshold_flat=0.36)(flat)["issues"] == []
