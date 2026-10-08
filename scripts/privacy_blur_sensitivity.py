"""How much does Panoramax's face/plate anonymisation lower the TOPIQ score?

Published Panoramax pictures have already been through SGBlur, which blurs
detected faces and plates inside JPEG blocks (hard-edged, block-aligned
rectangles; the rest of the picture is untouched). This script adds more
SGBlur-like patches to held-out pictures and measures the score drop:

* small: 6 distant faces / plates (1.5-3 % of the picture width each)
* large: 1 close pedestrian (12-20 % of the width)
* large2: 2 close pedestrians

    python scripts/privacy_blur_sensitivity.py data/eval --n 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from panomoche.imaging import load_image, prepare
from panomoche.iqa import TOPIQ, IQAScorer
from panomoche.panoramax import read_jsonl

MCU = 16  # SGBlur works on JPEG minimum coded units


def add_patches(img: Image.Image, boxes: list[tuple[float, float, float]], rng: np.random.Generator) -> Image.Image:
    """Blur block-aligned rectangles. ``boxes`` are (relative width, aspect h/w, relative x centre)."""
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    for rel_w, aspect, cx in boxes:
        bw = max(MCU, int(rel_w * w) // MCU * MCU)
        bh = max(MCU, int(bw * aspect) // MCU * MCU)
        x0 = int(np.clip(cx * w - bw / 2, 0, w - bw)) // MCU * MCU
        # faces and plates sit around the horizon: middle 40-65 % of the height
        y0 = int(rng.uniform(0.40, 0.65) * h - bh / 2) // MCU * MCU
        y0 = int(np.clip(y0, 0, h - bh))
        region = a[y0 : y0 + bh, x0 : x0 + bw]
        a[y0 : y0 + bh, x0 : x0 + bw] = cv2.GaussianBlur(region, (0, 0), max(4.0, 0.25 * min(bw, bh)))
    return Image.fromarray(a)


def variants(rng: np.random.Generator) -> dict[str, list]:
    return {
        "small": [(rng.uniform(0.015, 0.03), rng.uniform(0.6, 1.3), rng.uniform(0.05, 0.95)) for _ in range(6)],
        "large": [(rng.uniform(0.12, 0.20), rng.uniform(0.8, 1.4), rng.uniform(0.15, 0.85))],
        "large2": [(rng.uniform(0.12, 0.20), rng.uniform(0.8, 1.4), cx) for cx in (0.3, 0.7)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("data/privacy_blur_sensitivity.jsonl"))
    a = ap.parse_args()

    meta = {r["id"]: r for r in read_jsonl(a.folder / "meta.jsonl")}
    paths = sorted(a.folder.glob("*.jpg"))
    rng = np.random.default_rng(0)
    paths = [paths[i] for i in rng.permutation(len(paths))[: a.n]]
    scorer = IQAScorer(metrics=(TOPIQ,))
    rows = []
    for k, path in enumerate(paths, 1):
        img = load_image(path)
        pano = meta.get(path.stem, {}).get("is_pano")
        base = scorer.score_picture(prepare(img, pano).views())[f"{TOPIQ}_min"]
        row = {"id": path.stem, "is_pano": prepare(img, pano).is_pano, "base": base}
        for name, boxes in variants(rng).items():
            row[name] = scorer.score_picture(prepare(add_patches(img, boxes, rng), pano).views())[f"{TOPIQ}_min"]
        rows.append(row)
        print(f"{k}/{len(paths)}", {kk: round(v, 3) for kk, v in row.items() if isinstance(v, float)}, flush=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    print("\nscore drop (base - blurred), TOPIQ worst view:")
    for name in ("small", "large", "large2"):
        d = np.array([r["base"] - r[name] for r in rows])
        print(
            f"  {name:<7} median {np.median(d):+.3f}  p90 {np.quantile(d, 0.9):+.3f}  max {d.max():+.3f}  "
            f"≥0.10 (sequence min_drop): {int((d >= 0.10).sum())}/{len(d)}"
        )


if __name__ == "__main__":
    main()
