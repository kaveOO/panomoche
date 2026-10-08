"""Image loading and normalisation, shared by training and inference.

Everything downstream (synthetic degradations, signal metrics, DINOv3) works on
*views* at a fixed resolution so that scores are comparable between a 2048 px
flat photo and a 360° equirectangular panorama:

* flat picture  -> one view, long side ``VIEW_SIZE``
* 360° picture  -> the horizon band (latitudes ±45°) split into four 90° views.
  The nadir/zenith are dropped on purpose: the car roof, helmet or pole always
  visible there is normal for 360° rigs and must not count as an obstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

VIEW_SIZE = 512
PANO_BAND = (0.25, 0.75)  # fraction of the equirectangular height kept
PANO_VIEWS = 4


@dataclass
class Prepared:
    """Picture at working resolution, before it is split into views.

    For panoramas ``rgb`` is the horizon band (``PANO_VIEWS * VIEW_SIZE`` wide),
    so that synthetic degradations can be applied once across all views.
    """

    rgb: np.ndarray  # HxWx3 uint8
    is_pano: bool

    def views(self) -> list[np.ndarray]:
        if not self.is_pano:
            return [self.rgb]
        return [np.ascontiguousarray(v) for v in np.array_split(self.rgb, PANO_VIEWS, axis=1)]


def load_image(src) -> Image.Image:
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def is_equirectangular(img: Image.Image) -> bool:
    w, h = img.size
    return abs(w / h - 2.0) < 0.02


def prepare(img: Image.Image, is_pano: bool | None = None) -> Prepared:
    """Crop/resize a picture to working resolution. Never upscales."""
    pano = is_equirectangular(img) if is_pano is None else is_pano
    if pano:
        w, h = img.size
        img = img.crop((0, round(h * PANO_BAND[0]), w, round(h * PANO_BAND[1])))
        target_w = PANO_VIEWS * VIEW_SIZE
        scale = min(1.0, target_w / img.width)
    else:
        scale = min(1.0, VIEW_SIZE / max(img.size))
    if scale < 1.0:
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(size, Image.Resampling.LANCZOS)
    return Prepared(np.asarray(img, dtype=np.uint8).copy(), pano)
