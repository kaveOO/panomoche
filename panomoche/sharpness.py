"""Sharpness: is enough of the picture in focus?

Blur removes fine detail first, so the ratio of fine detail (``|I - G(1 px)|``)
to medium-scale detail (``|I - G(4 px)|``) drops on blurred areas whatever
their content. Measured on a 2048 px wide version (Panoramax "sd" size), on
the horizon band for 360° pictures, ignoring the boxes Panoramax blurred for
privacy.

A picture is judged on its *blurred area*: the share of its textured 64 px
blocks whose ratio is below 0.25. Flat blocks (sky, plain walls) have no
detail to judge and are left out. A whole-picture average was tried first and
missed partly blurred pictures: on a handlebar camera, a sharp distant street
in the middle hid motion blur over the whole foreground (Sony #303: average
0.31, but 53 % of its textured area blurred).

Measured (``scripts/sharpness_eval.py``): blurred copies of 60 held-out
pictures are separated from the originals with AUC 1.00 (mild and strong
Gaussian blur) and 0.91 (motion blur) by the detail ratio, against 0.60-0.78
for TOPIQ. On 600 real pictures, the 21 pictures with half or more of their
textured area blurred (21; 30 at the 0.45 default) are all handlebar-phone shots with motion blur and
wobble, or dashcam shots through a hazy windshield; 360° cameras have a median
blurred area of 4 %.

The detail ratio alone mistakes soft but clean textures (foliage, small
phone-app pictures, dusk) for blur. **CPBD** (Cumulative Probability of Blur
Detection, Narvekar & Karam 2011) is therefore used as a second opinion: it
measures the width of every vertical edge (Marziliano: the span of the
monotonic intensity run across it) and gives the share of edges whose blur
would go unnoticed by a viewer (edge width up to 5 px at low contrast, 3 px at
high contrast). It only looks at edges, so plain or soft textures don't count.
A picture is not sharp when both measures agree (``is_not_sharp``).

Measured (``scripts/cpbd_eval.py``): blurred copies of 60 held-out pictures are
separated with AUC 1.00 by CPBD for mild, strong and motion blur (detail ratio:
1.00 / 1.00 / 0.93). Alone, CPBD calls clean foliage and church pictures
blurry; as a confirmation it keeps 8 of the 35 pictures the detail ratio
excluded that look clean (a windmill, a chapel, sunsets, a 360° street), and
the second branch catches 17 handlebar pictures with rolling-shutter smear,
a washed-out van and a picture through a dirty side window that the detail
ratio alone let through.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .imaging import PANO_BAND

WORK_WIDTH = 1024  # judge blur at a viewing size: pixel-level softness at 100% zoom is not "unclean"
BLOCK = 64  # px, at 2048 px width
BLOCK_BLURRED = 0.25  # block detail ratio under which a block counts as blurred
FLAT_BLOCK = 3.0  # mean medium-scale detail under which a block is flat (sky, plain wall): not judged
DEFAULT_MAX_BLURRED_AREA = 0.33
DEFAULT_CONFIRM_CPBD = 0.15  # ...counts only if CPBD confirms (below this)
DEFAULT_SOFT_BLURRED_AREA = 0.20  # a smaller blurred area counts when CPBD is clearly low...
DEFAULT_SOFT_CPBD = 0.12  # ...below this
EDGE_BLOCK = 0.002  # share of edge pixels for a block to count in CPBD (paper)
JNB_CONTRAST = 50  # block contrast up to which the just-noticeable blur width is 5 px, else 3 px


def _prepare(
    img: Image.Image, is_pano: bool, mask_boxes, original_width: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    # Measure at the original resolution when it is smaller than the file: Panoramax's "sd" asset is
    # always 2048 px wide, so a 1,050 px original arrives enlarged twofold, without fine pixel detail,
    # and would look blurry however sharp it is (22% of a random Panoramax sample).
    target = min(WORK_WIDTH, original_width or WORK_WIDTH)
    if img.width > target:
        img = img.resize((target, round(img.height * target / img.width)), Image.Resampling.LANCZOS)
    g = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = g.shape
    keep = np.ones_like(g, bool)
    for x0, y0, x1, y1 in mask_boxes or ():
        keep[int(y0 * H) : int(np.ceil(y1 * H)), int(x0 * W) : int(np.ceil(x1 * W))] = False
    if is_pano:
        top, bottom = int(H * PANO_BAND[0]), int(H * PANO_BAND[1])
        g, keep = g[top:bottom], keep[top:bottom]
    return g, keep


def _run_lengths(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive-True counts along rows: ending at each index, and starting at each index."""
    idx = np.arange(a.shape[1])
    ending = idx - np.maximum.accumulate(np.where(a, -1, idx), axis=1)
    starting = (idx - np.maximum.accumulate(np.where(a[:, ::-1], -1, idx), axis=1))[:, ::-1]
    return ending, starting


def edge_widths(g: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marziliano widths of the vertical edges: (ys, xs, widths)."""
    u8 = np.clip(cv2.GaussianBlur(g, (0, 0), 1.0), 0, 255).astype(np.uint8)
    edges = cv2.Canny(u8, 50, 150) > 0
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    edges &= (np.abs(gx) > np.abs(gy)) & keep
    edges[:, 0] = edges[:, -1] = False
    inc, dec = g[:, 1:] > g[:, :-1], g[:, 1:] < g[:, :-1]  # diff i is between pixels i and i + 1
    inc_end, inc_start = _run_lengths(inc)
    dec_end, dec_start = _run_lengths(dec)
    ys, xs = np.nonzero(edges)
    rising = gx[ys, xs] > 0
    w = np.where(rising, inc_end[ys, xs - 1] + inc_start[ys, xs], dec_end[ys, xs - 1] + dec_start[ys, xs])
    ok = w > 0
    return ys[ok], xs[ok], w[ok].astype(np.float32)


def cpbd(g: np.ndarray, keep: np.ndarray) -> float | None:
    """Share of edges, in edge blocks, whose blur a viewer would not notice (1 = sharp)."""
    ys, xs, w = edge_widths(g, keep)
    cols = g.shape[1] // BLOCK + 1
    block_id = (ys // BLOCK) * cols + xs // BLOCK
    sharp = []
    for b in np.unique(block_id):
        sel = block_id == b
        if sel.sum() < EDGE_BLOCK * BLOCK * BLOCK:
            continue
        y0, x0 = (b // cols) * BLOCK, (b % cols) * BLOCK
        patch = g[y0 : y0 + BLOCK, x0 : x0 + BLOCK]
        w_jnb = 5 if patch.max() - patch.min() <= JNB_CONTRAST else 3
        sharp.append(1 - np.exp(-((w[sel] / w_jnb) ** 3.6)) <= 0.63)
    return round(float(np.concatenate(sharp).mean()), 4) if sharp else None


def measure(img: Image.Image, is_pano: bool, mask_boxes=(), original_width: int | None = None) -> dict:
    """``sharpness``: whole-picture detail ratio (for information);
    ``blurred_area``: share of the textured area made of blurred blocks;
    ``cpbd``: share of edges that look sharp (both used to exclude, see ``is_not_sharp``).

    ``mask_boxes``: relative [x0, y0, x1, y1] boxes to ignore (SGBlur's privacy blur).
    """
    g, keep = _prepare(img, is_pano, mask_boxes, original_width)
    if not keep.any():
        return {"sharpness": None, "blurred_area": None, "cpbd": None}
    fine = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.0))
    medium = np.abs(g - cv2.GaussianBlur(g, (0, 0), 4.0))
    blocks = []
    for y in range(0, g.shape[0] - BLOCK + 1, BLOCK):
        for x in range(0, g.shape[1] - BLOCK + 1, BLOCK):
            if keep[y : y + BLOCK, x : x + BLOCK].mean() < 0.9:
                continue
            m = medium[y : y + BLOCK, x : x + BLOCK].mean()
            if m >= FLAT_BLOCK:
                blocks.append(fine[y : y + BLOCK, x : x + BLOCK].mean() / m)
    return {
        "sharpness": round(float(fine[keep].mean() / (medium[keep].mean() + 1e-6)), 4),
        "blurred_area": round(float((np.array(blocks) < BLOCK_BLURRED).mean()), 4) if blocks else None,
        "cpbd": cpbd(g, keep),
    }


def is_not_sharp(
    blurred_area: float | None,
    max_blurred_area: float | None,
    cpbd: float | None = None,
    confirm_cpbd: float = DEFAULT_CONFIRM_CPBD,
    soft_blurred_area: float = DEFAULT_SOFT_BLURRED_AREA,
    soft_cpbd: float = DEFAULT_SOFT_CPBD,
) -> bool:
    """Both measures agree: a large blurred area that CPBD confirms, or a smaller one with a clearly low CPBD.

    Without CPBD (no edges, or not measured) the blurred area decides alone. A limit of 1 or more disables the rule.
    """
    if max_blurred_area is None or not 0 < max_blurred_area < 1 or blurred_area is None:
        return False
    if cpbd is None:
        return blurred_area >= max_blurred_area
    return (blurred_area >= max_blurred_area and cpbd < confirm_cpbd) or (
        blurred_area >= soft_blurred_area and cpbd < soft_cpbd
    )
