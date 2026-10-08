"""The single decision, checked on the reviewer's examples (measurements as observed)."""

import pytest

from panomoche.decision import Rules, decide


def picture(i, collection="seq", rank=None, topiq=0.6, blurred=0.05, **kw):
    return {
        "id": f"{collection}-{i}",
        "collection": collection,
        "rank": i if rank is None else rank,
        "quality_score": topiq,
        "blurred_area": blurred,
        "is_pano": False,
        "issues": [],
        **kw,
    }


def sequence(n, collection, **typical):
    return [picture(i, collection, **typical) for i in range(1, n + 1)]


def status(r):
    return r.get("excluded") or "kept"


def test_dashcam_through_hazy_windshield_is_kept():
    """Samsung #53: windshield haze (14% blurred at viewing size), TOPIQ 0.42 like its sequence."""
    rows = sequence(21, "dashcam", topiq=0.40, blurred=0.11)
    rows[10].update(quality_score=0.42, blurred_area=0.14)
    decide(rows)
    assert status(rows[10]) == "kept"


def test_same_picture_alone_is_kept_too():
    """Judged without its sequence, it gets the benefit of the doubt (flat floor 0.40)."""
    alone = picture(1, collection=None, topiq=0.42, blurred=0.14)
    assert status(decide([alone])[0]) == "kept"
    assert status(decide([picture(1, collection=None, topiq=0.33)])[0]) == "low_quality"  # night shot


def test_handlebar_picture_more_blurred_than_its_neighbours_is_excluded():
    """Sony #303: 35% of its textured area blurred at viewing size (neighbours: 28%)."""
    rows = sequence(21, "sony", topiq=0.58, blurred=0.28)
    rows[10].update(quality_score=0.61, blurred_area=0.35)
    assert status(decide(rows)[10]) == "not_sharp"


def test_glare_picture_much_worse_than_its_few_neighbours_is_excluded():
    """Forest in sun glare: TOPIQ 0.43 where the rest of its short sequence is at 0.63."""
    rows = [picture(i, "glare", topiq=0.63, blurred=0.04) for i in range(1, 6)]
    rows[3].update(quality_score=0.43, blurred_area=0.13)
    r = decide(rows)[3]
    assert status(r) == "sequence_outlier" and r["issues"] == ["sequence_outlier"] and r["low_quality"]


def test_dusk_picture_worse_than_its_neighbours_is_excluded():
    rows = [picture(i, "dusk", topiq=0.74, blurred=0.01) for i in range(1, 4)]
    rows[2].update(quality_score=0.60, blurred_area=0.16)
    assert status(decide(rows)[2]) == "sequence_outlier"


def test_soft_360_picture_in_line_with_its_sequence_is_kept():
    """GoPro Fusion #227."""
    rows = sequence(21, "fusion", topiq=0.49, blurred=0.0, is_pano=True)
    rows[10].update(quality_score=0.42, blurred_area=0.02)
    assert status(decide(rows)[10]) == "kept"


@pytest.mark.parametrize(
    "extra,reason",
    [
        ({"pixel_density": 11}, "low_resolution"),
        ({"privacy_blur": 0.09}, "privacy_blur"),
        ({"orientation": 180, "orientation_confidence": 0.999}, "wrong_orientation"),
        ({"blurred_area": 0.65}, "not_sharp"),
        ({"blurred_area": 0.40, "cpbd": 0.08}, "not_sharp"),
        ({"blurred_area": 0.25, "cpbd": 0.05}, "not_sharp"),
    ],
)
def test_hard_exclusions_apply_whatever_the_sequence(extra, reason):
    rows = sequence(21, "s", topiq=0.6, blurred=0.05)
    rows[10].update(extra)
    assert status(decide(rows)[10]) == reason


def test_soft_but_clean_texture_is_kept_when_edges_look_sharp():
    """Windmill, chapel, sunsets: 33-38% of the texture soft, but CPBD finds the edges sharp."""
    rows = [picture(1, None, topiq=0.6, blurred=0.36, cpbd=0.25)]
    assert status(decide(rows)[0]) != "not_sharp"


def test_topiq_floors():
    assert decide([picture(1, None, topiq=0.38)])[0]["issues"] == ["low_quality"]  # flat alone: 0.40
    assert decide([picture(1, None, topiq=0.42)])[0]["issues"] == []  # dull but clean road
    assert decide([picture(1, None, topiq=0.40, is_pano=True)])[0]["issues"] == []  # 360° alone: 0.32
    rows = sequence(5, "low", topiq=0.30)
    assert all(r["issues"] == ["low_quality"] for r in decide(rows))  # with neighbours: 0.32


def test_neighbours_follow_rank_and_window():
    rows = sequence(30, "s", topiq=0.6, blurred=0.05)
    rows[0].update(blurred_area=0.40)  # blurred, but far from rank 25
    rows[24].update(quality_score=0.40)
    decide(rows, Rules(window=10))
    assert status(rows[24]) == "sequence_outlier" and status(rows[0]) == "not_sharp"
    assert rows[24]["context"]["neighbours"] == 15  # ranks 15-35 that exist, minus itself


def test_decision_is_repeatable():
    rows = sequence(10, "s")
    rows[4].update(quality_score=0.3)
    first = [status(r) for r in decide(rows)]
    assert [status(r) for r in decide(rows)] == first


def test_mostly_sharp_picture_is_kept_even_if_neighbours_are_sharper():
    """360° pictures at 16-30% blurred next to very sharp neighbours were wrongly excluded."""
    rows = sequence(21, "lg", topiq=0.6, blurred=0.03, is_pano=True)
    rows[10]["blurred_area"] = 0.25
    assert status(decide(rows)[10]) == "kept"
