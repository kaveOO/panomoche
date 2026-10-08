import pytest

from panomoche.metadata import metadata_exclusion
from panomoche.panoramax import item_record


@pytest.mark.parametrize(
    "record,reason",
    [
        ({"gps_accuracy": 15.0, "pixel_density": 40}, "gps_inaccurate"),
        ({"gps_accuracy": 4.0, "pixel_density": 11}, "low_resolution"),
        ({"gps_accuracy": 4.0, "pixel_density": 16}, None),  # normal 360° picture
        ({"gps_accuracy": None, "pixel_density": None}, None),  # unknown never excludes
        ({}, None),
    ],
)
def test_metadata_rules(record, reason):
    assert metadata_exclusion(record, max_gps_accuracy=10) == reason


def test_gps_accuracy_is_kept_by_default():
    assert metadata_exclusion({"gps_accuracy": 36.0, "pixel_density": 58}) is None


def test_rules_can_be_disabled():
    assert metadata_exclusion({"gps_accuracy": 50, "pixel_density": 2}, 0, 0) is None


def test_item_record_reads_panoramax_quality_fields():
    item = {
        "id": "x",
        "links": [],
        "properties": {"quality:horizontal_accuracy": 4.0, "panoramax:horizontal_pixel_density": 16},
    }
    r = item_record(item)
    assert r["gps_accuracy"] == 4.0 and r["pixel_density"] == 16
