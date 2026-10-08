"""Does CPBD (Cumulative Probability of Blur Detection) judge sharpness better than the detail ratio?

CPBD (Narvekar & Karam, 2011) measures the width of every vertical edge (Marziliano: the span of the
monotonic intensity run across the edge) and compares it to the "just noticeable blur" width, 5 px at
low local contrast and 3 px at high contrast. CPBD = share of edges whose blur would go unnoticed
(1 = sharp). Unlike the detail ratio it only looks at edges, so plain or soft textures shouldn't count.

Measured the way ``sharpness.measure`` works (at min(1024, original) width, privacy boxes masked, 360°
horizon band), plus a block version: ``cpbd_area`` = share of edge blocks whose own CPBD is low, and the
variance of the Laplacian as a baseline.

1. synthetic: AUC original vs blurred copies (mild, strong, motion) on 60 held-out pictures;
2. real: every picture of ``data/all.jsonl`` -> ``data/cpbd_all.jsonl``.

    python scripts/cpbd_eval.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from panomoche.imaging import is_equirectangular, load_image
from panomoche.panoramax import read_jsonl
from panomoche.sharpness import BLOCK, _prepare, measure

EDGE_BLOCK = 0.002  # share of edge pixels for a block to count as an edge block (paper)
JNB_CONTRAST = 50  # block contrast up to which the just-noticeable blur width is 5 px, else 3 px
BLOCK_SHARP = 0.5  # block CPBD under which the block counts as blurred (for cpbd_area)


def _run_lengths(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive-True counts along rows: ending at each index, and starting at each index."""
    idx = np.arange(a.shape[1])
    ending = idx - np.maximum.accumulate(np.where(a, -1, idx), axis=1)
    r = a[:, ::-1]
    starting = (idx - np.maximum.accumulate(np.where(r, -1, idx), axis=1))[:, ::-1]
    return ending, starting


def edge_widths(g: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marziliano widths of vertical edges: (ys, xs, widths)."""
    u8 = np.clip(cv2.GaussianBlur(g, (0, 0), 1.0), 0, 255).astype(np.uint8)
    edges = cv2.Canny(u8, 50, 150) > 0
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    edges &= (np.abs(gx) > np.abs(gy)) & keep
    edges[:, 0] = edges[:, -1] = False
    inc, dec = g[:, 1:] > g[:, :-1], g[:, 1:] < g[:, :-1]  # diff i is between pixels i and i+1
    inc_end, inc_start = _run_lengths(inc)
    dec_end, dec_start = _run_lengths(dec)
    ys, xs = np.nonzero(edges)
    rising = gx[ys, xs] > 0
    w = np.where(rising, inc_end[ys, xs - 1] + inc_start[ys, xs], dec_end[ys, xs - 1] + dec_start[ys, xs])
    ok = w > 0
    return ys[ok], xs[ok], w[ok].astype(np.float32)


def cpbd(img: Image.Image, is_pano: bool, mask_boxes=(), original_width=None) -> dict:
    g, keep = _prepare(img, is_pano, mask_boxes, original_width)
    ys, xs, w = edge_widths(g, keep)
    by, bx = ys // BLOCK, xs // BLOCK
    nbx = g.shape[1] // BLOCK
    block_id = by * (nbx + 1) + bx
    sharp_all, block_scores = [], []
    for b in np.unique(block_id):
        sel = block_id == b
        if sel.sum() < EDGE_BLOCK * BLOCK * BLOCK:
            continue
        y0, x0 = (b // (nbx + 1)) * BLOCK, (b % (nbx + 1)) * BLOCK
        patch = g[y0 : y0 + BLOCK, x0 : x0 + BLOCK]
        w_jnb = 5 if patch.max() - patch.min() <= JNB_CONTRAST else 3
        p_blur = 1 - np.exp(-((w[sel] / w_jnb) ** 3.6))
        sharp = p_blur <= 0.63
        sharp_all.append(sharp)
        block_scores.append(sharp.mean())
    lap = cv2.Laplacian(g, cv2.CV_32F)[keep]
    return {
        "cpbd": round(float(np.concatenate(sharp_all).mean()), 4) if sharp_all else None,
        "cpbd_area": round(float((np.array(block_scores) < BLOCK_SHARP).mean()), 4) if block_scores else None,
        "edge_blocks": len(block_scores),
        "lapvar": round(float(np.log(lap.var() + 1e-6)), 4) if lap.size else None,
    }


def jpeg(img: Image.Image, q: int = 85) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def motion(a: np.ndarray, length: int) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2] = 1.0 / length
    return cv2.filter2D(a, -1, k)


def synthetic(n: int = 60) -> None:
    rng = np.random.default_rng(0)
    paths = sorted(Path("data/eval").glob("*.jpg"))
    paths = [paths[i] for i in rng.permutation(len(paths))[:n]]
    rows = []
    for path in paths:
        img = load_image(path)
        if img.width > 1024:
            img = img.resize((1024, round(img.height * 1024 / img.width)), Image.Resampling.LANCZOS)
        pano = is_equirectangular(img)
        a = np.asarray(img)
        variants = {
            "original": a,
            "mild": cv2.GaussianBlur(a, (0, 0), 1.0),
            "strong": cv2.GaussianBlur(a, (0, 0), 2.0),
            "motion": motion(a, 7),
        }
        for name, v in variants.items():
            im = jpeg(Image.fromarray(v))
            m = cpbd(im, pano) | measure(im, pano)
            rows.append(
                {
                    "variant": name,
                    "cpbd": m["cpbd"],
                    "-cpbd_area": -(m["cpbd_area"] or 0),
                    "lapvar": m["lapvar"],
                    "-blurred_area": -(m["blurred_area"] or 0),
                }
            )
    metrics = [k for k in rows[0] if k != "variant"]
    print(f"AUC, original vs blurred copy, {n} pictures at 1024 px (1.0 = perfect, 0.5 = chance):")
    print(f"  {'blur':<8}" + "".join(f"{m:>15}" for m in metrics))
    for blur in ("mild", "strong", "motion"):
        sel = [r for r in rows if r["variant"] in ("original", blur)]
        y = [r["variant"] == "original" for r in sel]
        print(f"  {blur:<8}" + "".join(f"{roc_auc_score(y, [r[m] for r in sel]):>15.2f}" for m in metrics))


def real() -> None:
    rows = read_jsonl(Path("data/all.jsonl"))
    t = time.time()
    with open("data/cpbd_all.jsonl", "w") as f:
        for r in rows:
            m = cpbd(
                load_image(r["path"]), r["is_pano"], r.get("privacy_boxes") or (), (r.get("original_size") or [None])[0]
            )
            f.write(json.dumps({"id": r["id"]} | m) + "\n")
    print(f"real: {len(rows)} pictures in {time.time() - t:.0f}s -> data/cpbd_all.jsonl")


if __name__ == "__main__":
    if "--real-only" not in sys.argv:
        synthetic()
    real()
