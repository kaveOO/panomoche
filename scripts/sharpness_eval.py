"""Can sharpness measures tell sharp pictures from blurred ones?

1. synthetic: blur 60 held-out pictures (mild, strong, motion), re-encode all
   versions alike, and measure how well each measure separates the blurred
   copies from the originals across different pictures and cameras (AUC);
2. real: compute the measures on the scored sequences and save them for a
   sequence-relative look (``data/sharpness_sequences.json``).

    python scripts/sharpness_eval.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from panomoche.imaging import PANO_BAND, is_equirectangular, load_image, prepare
from panomoche.iqa import TOPIQ, IQAScorer
from panomoche.panoramax import read_jsonl

WORK = 2048
BLOCK = 64


def sharpness(img: Image.Image, is_pano: bool, mask_boxes=()) -> dict:
    """Sharpness measures at 2048 px, on the horizon band for 360° pictures, ignoring privacy-blur boxes."""
    if img.width > WORK:
        img = img.resize((WORK, round(img.height * WORK / img.width)), Image.Resampling.LANCZOS)
    g = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = g.shape
    keep = np.ones_like(g, bool)
    for x0, y0, x1, y1 in mask_boxes:
        keep[int(y0 * H) : int(y1 * H), int(x0 * W) : int(x1 * W)] = False
    if is_pano:
        t, b = int(H * PANO_BAND[0]), int(H * PANO_BAND[1])
        g, keep = g[t:b], keep[t:b]
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    grad2 = gx * gx + gy * gy
    fine = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.0))
    medium = np.abs(g - cv2.GaussianBlur(g, (0, 0), 4.0))
    blocks_t, blocks_r = [], []
    for y in range(0, g.shape[0] - BLOCK + 1, BLOCK):
        for x in range(0, g.shape[1] - BLOCK + 1, BLOCK):
            if keep[y : y + BLOCK, x : x + BLOCK].mean() < 0.9:
                continue
            m = medium[y : y + BLOCK, x : x + BLOCK].mean()
            blocks_t.append(grad2[y : y + BLOCK, x : x + BLOCK].mean())
            if m > 2.0:  # skip flat blocks (sky, plain walls): no detail to judge
                blocks_r.append(fine[y : y + BLOCK, x : x + BLOCK].mean() / m)
    return {
        "tenengrad": float(np.log(grad2[keep].mean() + 1e-6)),
        "fine_medium": float(fine[keep].mean() / (medium[keep].mean() + 1e-6)),
        "best_tenengrad": float(np.log(np.quantile(blocks_t, 0.9) + 1e-6)) if blocks_t else float("nan"),
        "best_fine_medium": float(np.quantile(blocks_r, 0.9)) if blocks_r else float("nan"),
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
    scorer = IQAScorer(metrics=(TOPIQ,))
    rows = []
    for path in paths:
        img = load_image(path)
        if img.width > WORK:
            img = img.resize((WORK, round(img.height * WORK / img.width)), Image.Resampling.LANCZOS)
        pano = is_equirectangular(img)
        a = np.asarray(img)
        variants = {
            "original": a,
            "mild": cv2.GaussianBlur(a, (0, 0), 1.0),
            "strong": cv2.GaussianBlur(a, (0, 0), 2.5),
            "motion": motion(a, 9),
        }
        for name, v in variants.items():
            im = jpeg(Image.fromarray(v))
            m = sharpness(im, pano)
            m["topiq"] = scorer.score_picture(prepare(im, pano).views())[f"{TOPIQ}_min"]
            rows.append({"variant": name, **m})
    print(f"AUC, original vs blurred copy, pooled over {n} pictures (1.0 = perfect, 0.5 = chance):")
    metrics = [k for k in rows[0] if k != "variant"]
    print(f"  {'blur':<8}" + "".join(f"{m:>18}" for m in metrics))
    for blur in ("mild", "strong", "motion"):
        sel = [r for r in rows if r["variant"] in ("original", blur)]
        y = [r["variant"] == "original" for r in sel]
        print(f"  {blur:<8}" + "".join(f"{roc_auc_score(y, [r[m] for r in sel]):>18.2f}" for m in metrics))


def real() -> None:
    rows = read_jsonl(Path("data/sequences.jsonl"))
    out = []
    for r in rows:
        m = sharpness(load_image(r["path"]), r["is_pano"], r.get("privacy_boxes") or ())
        out.append(
            {
                k: r.get(k)
                for k in (
                    "id",
                    "collection",
                    "rank",
                    "camera",
                    "is_pano",
                    "quality_score",
                    "issues",
                    "excluded",
                    "path",
                )
            }
            | m
        )
    Path("data/sharpness_sequences.json").write_text(json.dumps(out))
    print(f"saved sharpness of {len(out)} sequence pictures")


if __name__ == "__main__":
    synthetic()
    real()
