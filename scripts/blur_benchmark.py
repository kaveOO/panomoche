"""Benchmark of the blur detection ("not_sharp": blurred area + CPBD, see sharpness.py).

1. Controlled blur: random pictures of the new dataset (never used to tune the rule) are brought
   to viewing size (1024 px), blurred (Gaussian focus blur, linear motion blur at a random angle),
   JPEG-encoded like Panoramax files, and measured as the pipeline does.
2. Each alternative measure gets the threshold that flags as many untouched pictures as the rule
   does, so methods are compared on blurred copies at the same false-alarm rate.
3. Speed.

    python scripts/blur_benchmark.py --n 200      # writes reports/blur_benchmark.{json,md}
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from panomoche.imaging import load_image, prepare
from panomoche.panoramax import read_jsonl
from panomoche.sharpness import is_not_sharp, measure

WIDTH = 1024
VARIANTS = [
    ("original", None, 0),
    ("gaussian", "σ 0.75 px", 0.75),
    ("gaussian", "σ 1.5 px", 1.5),
    ("gaussian", "σ 3 px", 3.0),
    ("motion", "5 px", 5),
    ("motion", "11 px", 11),
    ("motion", "21 px", 21),
]


def motion_kernel(length: int, angle: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2] = 1.0
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0), (length, length))
    return k / k.sum()


def degrade(img: Image.Image, kind: str | None, amount: float, rng) -> Image.Image:
    a = np.asarray(img)
    if kind == "gaussian":
        a = cv2.GaussianBlur(a, (0, 0), amount)
    elif kind == "motion":
        a = cv2.filter2D(a, -1, motion_kernel(int(amount), rng.uniform(0, 180)))
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=90)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def lapvar(img: Image.Image, is_pano: bool) -> float:
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    if is_pano:
        g = g[g.shape[0] // 4 : 3 * g.shape[0] // 4]
    return float(np.log(cv2.Laplacian(g, cv2.CV_32F).var() + 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-topiq", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    pool = read_jsonl(Path("data/new.jsonl"))
    sample = [pool[i] for i in rng.permutation(len(pool))[: a.n]]
    topiq = None
    if not a.no_topiq:
        from panomoche.scoring import make_scorer

        topiq, _ = make_scorer("topiq", threads=10)

    rows, timing = [], {"blur rule (area + CPBD)": [], "Laplacian variance": [], "TOPIQ": []}
    for r in sample:
        img = load_image(r["path"])
        if img.width > WIDTH:
            img = img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.Resampling.LANCZOS)
        for kind, level, amount in VARIANTS:
            v = degrade(img, kind, amount, rng)
            t = time.perf_counter()
            m = measure(v, r["is_pano"], (), v.width)
            timing["blur rule (area + CPBD)"].append(time.perf_counter() - t)
            t = time.perf_counter()
            lv = lapvar(v, r["is_pano"])
            timing["Laplacian variance"].append(time.perf_counter() - t)
            tq = None
            if topiq:
                t = time.perf_counter()
                tq = topiq(prepare(v, r["is_pano"]))["quality_score"]
                timing["TOPIQ"].append(time.perf_counter() - t)
            rows.append(
                {
                    "id": r["id"],
                    "camera": r.get("camera"),
                    "is_pano": r["is_pano"],
                    "kind": kind or "original",
                    "level": level or "untouched",
                    "blurred_area": m["blurred_area"],
                    "cpbd": m["cpbd"],
                    "lapvar": lv,
                    "topiq": tq,
                }
            )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/blur_benchmark_rows.json").write_text(json.dumps(rows))

    # Detection: "higher = blurrier" score per method; rule is a fixed decision.
    def rule(x):
        return is_not_sharp(x["blurred_area"], 0.33, x["cpbd"])

    scores = {
        "blurred area alone": lambda x: x["blurred_area"] if x["blurred_area"] is not None else 1.0,
        "CPBD alone": lambda x: -(x["cpbd"] if x["cpbd"] is not None else 0.0),
        "Laplacian variance": lambda x: -x["lapvar"],
    }
    if topiq:
        scores["TOPIQ"] = lambda x: -x["topiq"]
    orig = [x for x in rows if x["kind"] == "original"]
    flag_rate = np.mean([rule(x) for x in orig])
    thresholds = {k: float(np.quantile([f(x) for x in orig], 1 - flag_rate)) for k, f in scores.items()}
    levels = [(kind, level) for kind, level, _ in VARIANTS]
    table = []
    for kind, level in levels:
        sel = [x for x in rows if x["kind"] == (kind or "original") and x["level"] == (level or "untouched")]
        line = {"blur": f"{kind or 'original'} {level or ''}".strip(), "rule": float(np.mean([rule(x) for x in sel]))}
        for k, f in scores.items():
            line[k] = float(np.mean([f(x) > thresholds[k] for x in sel]))
        table.append(line)
    speed = {k: round(1000 * float(np.median(v)), 1) for k, v in timing.items() if v}

    out = {
        "pictures": len(sample),
        "panoramas": sum(r["is_pano"] for r in sample),
        "cameras": len({r.get("camera") for r in sample}),
        "flag_rate_untouched": float(flag_rate),
        "thresholds": thresholds,
        "detection": table,
        "median_ms": speed,
    }
    Path("reports/blur_benchmark.json").write_text(json.dumps(out, indent=1))
    cols = ["rule"] + list(scores)
    md = [
        f"Pictures: {out['pictures']} ({out['panoramas']} 360°, {out['cameras']} cameras) from data/new.jsonl, "
        f"each untouched and blurred 6 ways. Untouched pictures flagged by the rule: {flag_rate:.1%} "
        "(the other methods are set to the same rate).",
        "",
        "| blur | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    md += [f"| {t['blur']} | " + " | ".join(f"{100 * t[c]:.0f}%" for c in cols) + " |" for t in table]
    md += [
        "",
        "Median time per picture: " + ", ".join(f"{k} {v} ms" for k, v in speed.items()),
    ]
    Path("reports/blur_benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
