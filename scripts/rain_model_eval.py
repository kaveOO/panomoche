"""Rain on the lens: a DINOv3 classifier trained on one rainy sequence, tested on another (prototype).

Rain drops are local: a few soft patches with a sharp scene around them, so the blur rule
(which averages over the picture) does not see them. Block-sharpness rules were tried (soft
patches that stay in place while the scene moves, with or without sharp surroundings) and did
not separate rainy from clean sequences; see docs/METHODS.md, "Rain on the lens".

Positives: frames of a rainy IGN GoPro Max sequence (``data/context_fb371d59.jsonl``).
Negatives: random clean GoPro Max pictures (same camera, so the model cannot learn the camera).
Test: the other rainy sequence (``data/rain_seq/frames.json``, never seen) and held-out clean
GoPro Max pictures.

    python scripts/rain_model_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from panomoche.backbone import DinoBackbone
from panomoche.imaging import load_image, prepare
from panomoche.panoramax import read_jsonl


def main():
    backbone = DinoBackbone(threads=10)

    def embed(paths):
        return np.array([backbone.embed_views(prepare(load_image(p), True).views()).mean(0) for p in paths])

    train_rain_rows = read_jsonl(Path("data/context_fb371d59.jsonl"))
    train_rain = [r["path"] for r in train_rain_rows]
    test_rain = [r["path"] for r in json.loads(Path("data/rain_seq/frames.json").read_text())]
    rainy_ids = {r["id"] for r in train_rain_rows}
    clean = [
        r
        for r in read_jsonl(Path("data/all.jsonl"))
        if r["camera"] == "GoPro Max" and r["is_pano"] and r["id"] not in rainy_ids
    ]
    order = np.random.default_rng(0).permutation(len(clean))
    test_clean = [clean[i]["path"] for i in order[:30]]
    train_clean = [clean[i]["path"] for i in order[30:]]

    X = np.vstack([embed(train_rain), embed(train_clean)])
    y = np.r_[np.ones(len(train_rain)), np.zeros(len(train_clean))]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))
    model.fit(X, y)
    p_rain = model.predict_proba(embed(test_rain))[:, 1]
    p_clean = model.predict_proba(embed(test_clean))[:, 1]
    print(f"trained on {len(train_rain)} rainy frames (one sequence) + {len(train_clean)} clean GoPro Max pictures")
    print(
        f"unseen rainy sequence: {(p_rain > 0.5).sum()}/{len(p_rain)} detected "
        f"(scores {p_rain.min():.2f}-{p_rain.max():.2f})"
    )
    print(f"unseen clean pictures: {(p_clean > 0.5).sum()}/{len(p_clean)} flagged (max score {p_clean.max():.2f})")
    auc = roc_auc_score(np.r_[np.ones(len(p_rain)), np.zeros(len(p_clean))], np.r_[p_rain, p_clean])
    print(f"AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
