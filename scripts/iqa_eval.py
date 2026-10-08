"""Score a folder of pictures with off-the-shelf IQA models, then write a review sheet.

    python scripts/iqa_eval.py data/eval --out data/iqa_eval.jsonl --html reports/iqa_eval.html

Resumable: pictures already in --out are skipped. A watchdog aborts the run if
memory goes above --max-gb, instead of letting the OOM killer pick a victim.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image


def watchdog(max_bytes: float) -> None:
    import psutil

    proc = psutil.Process()
    while True:
        if proc.memory_info().rss > max_bytes:
            print(f"ABORT: memory above {max_bytes / 2**30:.1f} GB", flush=True)
            os._exit(3)
        time.sleep(0.1)


def score(folder: Path, out: Path, threads: int) -> None:
    import torch

    from panomoche.imaging import load_image, prepare
    from panomoche.iqa import IQAScorer
    from panomoche.panoramax import read_jsonl

    torch.set_num_threads(threads)
    meta = {r["id"]: r for r in read_jsonl(folder / "meta.jsonl")} if (folder / "meta.jsonl").exists() else {}
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [p for p in sorted(folder.glob("*.jpg")) if p.stem not in done]
    print(f"{len(done)} already scored, {len(todo)} to go", flush=True)
    scorer = IQAScorer()
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, "a") as f:
        for k, path in enumerate(todo, 1):
            m = meta.get(path.stem, {})
            try:
                prep = prepare(load_image(path), m.get("is_pano"))
            except Exception as exc:
                print(f"skip {path.name}: {exc}", flush=True)
                continue
            res = scorer.score_picture(prep.views())
            f.write(
                json.dumps(
                    {"id": path.stem, "path": str(path), "is_pano": prep.is_pano, "camera": m.get("camera"), **res}
                )
                + "\n"
            )
            f.flush()
            if k % 20 == 0 or k == len(todo):
                rate = (time.time() - t0) / k
                print(f"{k}/{len(todo)}  {rate:.1f} s/pic  ~{rate * (len(todo) - k) / 60:.0f} min left", flush=True)


def _percentile(values: np.ndarray) -> np.ndarray:
    return values.argsort().argsort() / max(1, len(values) - 1)


def _thumb(path: str, width: int = 360) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((width, width))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


CSS = """
:root{color-scheme:light dark;--bg:#f6f6f4;--card:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e4e0;--bad:#c2410c;--ok:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#21211f;--fg:#ececea;--muted:#9a9a94;--line:#33332f;--bad:#fb923c;--ok:#4ade80}}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:20px;margin:0 0 4px}p{color:var(--muted);margin:0 0 12px;max-width:80ch}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.card img{width:100%;aspect-ratio:3/2;object-fit:cover;display:block;background:var(--line)}
.m{padding:8px 10px;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.rank{font-weight:600;color:var(--fg);font-size:13px}.tag{display:inline-block;padding:0 6px;border-radius:4px;margin:2px 4px 2px 0;
background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}.tag.ok{background:color-mix(in srgb,var(--ok) 16%,transparent);color:var(--ok)}
table{border-collapse:collapse;width:100%}td{padding:1px 0}td:last-child{text-align:right}
"""


def report(scored: Path, out: Path) -> None:
    from panomoche.panoramax import read_jsonl

    rows = read_jsonl(scored)
    keys = ("topiq_nr-spaq", "musiq-spaq", "liqe_mix")
    pct = np.mean([_percentile(np.array([r[f"{k}_min"] for r in rows])) for k in keys], axis=0)
    for r, p in zip(rows, pct):
        r["consensus"] = float(p)
    rows.sort(key=lambda r: r["consensus"])
    cards = []
    for i, r in enumerate(rows, 1):
        dist = max(r["liqe_distortion"], key=r["liqe_distortion"].get)
        scene = max(r["liqe_scene"], key=r["liqe_scene"].get)
        tags = f'<span class="tag">{html.escape(dist)} {r["liqe_distortion"][dist]:.0%}</span>'
        tags += f'<span class="tag ok">{html.escape(scene)}</span>'
        cards.append(
            f'<div class="card"><img loading="lazy" src="{_thumb(r["path"])}" alt="">'
            f'<div class="m"><div class="rank">#{i} · {"360°" if r["is_pano"] else "flat"} · '
            f"{html.escape(r.get('camera') or 'unknown camera')}</div>{tags}<table>"
            f"<tr><td>TOPIQ (0–1)</td><td>{r['topiq_nr-spaq']:.2f} / worst view {r['topiq_nr-spaq_min']:.2f}</td></tr>"
            f"<tr><td>MUSIQ (0–100)</td><td>{r['musiq-spaq']:.0f} / {r['musiq-spaq_min']:.0f}</td></tr>"
            f"<tr><td>LIQE (1–5)</td><td>{r['liqe_mix']:.2f} / {r['liqe_mix_min']:.2f}</td></tr>"
            f"</table><div>{html.escape(r['id'])}</div></div></div>"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>IQA eval review</title>'
        f"<style>{CSS}</style></head><body><h1>Off-the-shelf IQA on {len(rows)} Panoramax pictures</h1>"
        "<p>Ranked worst first by the consensus of three pretrained models (TOPIQ-NR and MUSIQ trained on SPAQ "
        "smartphone photos, LIQE-mix), using each picture's worst view. Orange tag: LIQE's most likely "
        "distortion. Green tag: LIQE's scene guess. No model was trained on Panoramax data.</p>"
        f'<div class="grid">{"".join(cards)}</div></body></html>'
    )
    print(f"review sheet: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/iqa_eval.jsonl"))
    ap.add_argument("--html", type=Path, default=Path("reports/iqa_eval.html"))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-gb", type=float, default=4.0)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    threading.Thread(target=watchdog, args=(a.max_gb * 2**30,), daemon=True).start()
    if not a.report_only:
        score(a.folder, a.out, a.threads)
    report(a.out, a.html)


if __name__ == "__main__":
    main()
