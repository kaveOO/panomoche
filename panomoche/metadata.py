"""Checks on the metadata Panoramax stores with each picture.

* ``quality:horizontal_accuracy``: GPS accuracy in metres, as reported by the
  camera or computed by Panoramax. Most pictures report 2-5 m. It is recorded
  but doesn't exclude pictures by default; ``--max-gps-accuracy 10`` would
  exclude positions worse than 10 m.
* ``panoramax:horizontal_pixel_density``: pixels per degree of field of view.
  A GoPro Max 360° picture is 16 px/° (5,760 px over 360°), so the minimum
  stays below that; it only catches very low-resolution pictures.

A missing value never excludes a picture (in a 700-picture random sample,
accuracy was known for 61% and density for 44%).
"""

from __future__ import annotations

DEFAULT_MAX_GPS_ACCURACY = 0.0  # metres; 0 = recorded but not used to exclude (user decision)
DEFAULT_MIN_PIXEL_DENSITY = 15  # px per degree


def metadata_exclusion(
    record: dict,
    max_gps_accuracy: float | None = DEFAULT_MAX_GPS_ACCURACY,
    min_pixel_density: float | None = DEFAULT_MIN_PIXEL_DENSITY,
) -> str | None:
    accuracy, density = record.get("gps_accuracy"), record.get("pixel_density")
    if max_gps_accuracy and accuracy is not None and accuracy > max_gps_accuracy:
        return "gps_inaccurate"
    if min_pixel_density and density is not None and density < min_pixel_density:
        return "low_resolution"
    return None
