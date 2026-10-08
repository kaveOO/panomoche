"""Orientation: is the picture the right way up?

Upside-down or sideways pictures happen when a camera is mounted inverted or
the phone's orientation sensor is wrong (Sony #305 and #348 of the handlebar
sequence are upside down, and TOPIQ rates #305 0.59, "fine").

No labels are needed: good pictures are rotated by 0, 90, 180 and 270 degrees
and a logistic regression learns to recognise the rotation from frozen DINOv3
features (``train``). A picture whose content looks rotated with high
confidence is excluded as ``wrong_orientation``.

360° pictures: a 360° camera mounted pitched or rolled by 90° (a helmet
camera on its side) puts the ground in the middle of the picture and the sky
on the sides. Four ordinary 90° camera views are rendered from the full
sphere (ahead, right, behind, left, on the horizon, ``perspective``) and
judged like flat pictures. A level camera gives four upright views; a pitched
one gives sideways views. The picture is excluded when at least two views look
rotated with the minimum confidence: one rotated view happens on clean
pictures (a close wall, a facade seen from below). On 491 360° pictures this
caught 8 helmet pictures of an LG sequence, all mounted 90° off, and no clean
picture; a ninth (LG #111) stays at 98% and is missed. The previous approach,
averaging the four horizon-band crops, gave these pictures about 50%.

Measured (``scripts/orientation_eval.py``): 96.6% of rotated copies of 200
held-out pictures recognised. On 1,300 real pictures, the 5 predicted rotated
with confidence 1.00 are all really upside down or sideways; the 4 at
0.97-0.98 were not (an aerial view, an information board, a view past a
camper's ladder, a tilted fisheye road), hence the 0.99 minimum.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image

from .imaging import Prepared

ROTATIONS = (0, 90, 180, 270)  # clockwise degrees needed to bring the picture back upright
SIZE = 336  # long side fed to DINOv3; orientation doesn't need more detail
DEFAULT_MODEL = Path("models/orientation.joblib")
DEFAULT_MIN_CONFIDENCE = 0.99
PANO_YAWS = (0, 90, 180, 270)  # 360° pictures: four horizon views, ahead / right / behind / left
MIN_ROTATED_VIEWS = 2  # ...of which this many must look rotated
PANO_WORK_WIDTH = 2048  # equirectangular width the views are rendered from


def _resize(view: np.ndarray) -> np.ndarray:
    img = Image.fromarray(view)
    scale = SIZE / max(img.size)
    return np.asarray(img.resize((max(16, round(img.width * scale)), max(16, round(img.height * scale)))))


def rotate(view: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a view counter-clockwise by ``degrees`` (what a mis-oriented camera produces)."""
    return np.ascontiguousarray(np.rot90(view, k=degrees // 90))


def perspective(equirect: np.ndarray, yaw: float, fov: float = 90, size: int = SIZE) -> np.ndarray:
    """Ordinary (rectilinear) camera view rendered from an equirectangular picture, looking at the horizon."""
    h, w = equirect.shape[:2]
    f = size / 2 / np.tan(np.radians(fov) / 2)
    u, v = np.meshgrid(np.arange(size) - size / 2 + 0.5, np.arange(size) - size / 2 + 0.5)
    x, y = u / f, -v / f
    lon = np.arctan2(x, 1.0) + np.radians(yaw)
    lat = np.arctan2(y, np.hypot(x, 1.0))
    map_x = (((lon / (2 * np.pi) + 0.5) % 1.0) * w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)
    return cv2.remap(equirect, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def pano_views(img: Image.Image) -> list[np.ndarray]:
    if img.width > PANO_WORK_WIDTH:
        img = img.resize((PANO_WORK_WIDTH, PANO_WORK_WIDTH // 2), Image.Resampling.LANCZOS)
    eq = np.asarray(img.convert("RGB"))
    return [perspective(eq, yaw) for yaw in PANO_YAWS]


def embed(backbone, views: list[np.ndarray]) -> np.ndarray:
    return backbone.embed_views([_resize(v) for v in views])


def train(backbone, preps: list[Prepared], seed: int = 0, C: float = 1.0):
    """Fit the 4-way rotation classifier on rotated copies of upright pictures."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    X, y = [], []
    for prep in preps:
        views = prep.views()
        view = views[rng.integers(len(views))]  # one view per picture keeps the set balanced and fast
        for k, deg in enumerate(ROTATIONS):
            X.append(embed(backbone, [rotate(view, deg)])[0])
            y.append(k)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    clf.fit(np.array(X), np.array(y))
    return clf


def predict(backbone, clf, prep: Prepared, image: Image.Image | None = None) -> dict:
    """``orientation``: rotation the content shows (0 = upright); ``orientation_confidence``: its probability.

    For a 360° picture, pass the full equirectangular ``image``: the four horizon views are judged
    one by one (``orientation_views``: [rotation, probability] per view); ``orientation`` is then
    the rotation of the second most confident rotated view, so that ``is_wrong`` needs two views.
    """
    if prep.is_pano and image is not None:
        probs = clf.predict_proba(embed(backbone, pano_views(image)))
        views = [[ROTATIONS[int(p.argmax())], round(float(p.max()), 4)] for p in probs]
        rotated = sorted((v for v in views if v[0] != 0), key=lambda v: -v[1])
        rot, conf = (
            rotated[MIN_ROTATED_VIEWS - 1] if len(rotated) >= MIN_ROTATED_VIEWS else (0, min(v[1] for v in views))
        )
        return {"orientation": rot, "orientation_confidence": conf, "orientation_views": views}
    probs = clf.predict_proba(embed(backbone, prep.views())).mean(0)
    k = int(probs.argmax())
    return {"orientation": ROTATIONS[k], "orientation_confidence": round(float(probs[k]), 4)}


def is_wrong(result: dict | None, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> bool:
    if result and result.get("orientation_views"):
        return (
            sum(rot != 0 and conf >= min_confidence for rot, conf in result["orientation_views"]) >= MIN_ROTATED_VIEWS
        )
    return (
        bool(result)
        and (result.get("orientation") or 0) != 0
        and (result.get("orientation_confidence") or 0) >= min_confidence
    )


def save(clf, path: Path = DEFAULT_MODEL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)


def load(path: Path = DEFAULT_MODEL):
    return joblib.load(path)
