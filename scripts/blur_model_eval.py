"""Can a classifier learn "blurred" from the reviewer's labels? (DINOv3 features + blur measures)

Positives: pictures marked Blurred on the review page. Negatives: pictures marked OK, plus kept
pictures from sequences the reviewer did not label at all (inside a labelled sequence, an
unlabelled picture may simply not have been reviewed). Pictures the rules already exclude are
left out. Tested by holding out whole sequences, so that the model cannot pass by recognising
the camera; the new dataset (``data/new.jsonl``) is never used for training.

    python scripts/blur_model_eval.py      # needs data/dino_embeddings.npz
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from panomoche.panoramax import read_jsonl

MEASURES = ("blurred_area", "cpbd", "quality_score", "is_pano")


def features(rows, emb):
    m = np.array([[float(r.get(k) or 0) for k in MEASURES] for r in rows], np.float32)
    return np.hstack([np.stack([emb[r["id"]] for r in rows]), m])


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))


def main():
    e = np.load("data/dino_embeddings.npz")
    emb = dict(zip(e["ids"], e["X"]))
    labels = json.loads(Path("data/user_labels.json").read_text())
    old = read_jsonl(Path("data/all.jsonl"))
    new = read_jsonl(Path("data/new.jsonl"))
    labelled_seqs = {r["collection"] for r in old if r["id"] in labels}
    train = [
        r
        for r in old
        if labels.get(r["id"]) in ("blurred", "ok")
        or (
            not r.get("rules_excluded", r["excluded"])
            and r["id"] not in labels
            and r.get("collection") not in labelled_seqs
        )
    ]
    y = np.array([labels.get(r["id"]) == "blurred" for r in train])
    X = features(train, emb)
    group = [r.get("collection") or r["id"] for r in train]
    print(f"training set: {y.sum()} blurred, {(~y).sum()} not blurred, {len(set(group))} sequences")

    # Out-of-group scores: each positive sequence (and each lone picture) is scored by a model
    # that never saw it; negatives are scored in 5 sequence-disjoint folds.
    oof = np.full(len(train), np.nan)
    pos_groups = sorted({g for g, t in zip(group, y) if t})
    for g in pos_groups:
        test = np.array([gg == g for gg in group])
        oof[test] = model().fit(X[~test], y[~test]).predict_proba(X[test])[:, 1]
    rest = np.array([g not in pos_groups for g in group])
    folds = {g: i % 5 for i, g in enumerate(sorted({g for g, r in zip(group, rest) if r}))}
    for k in range(5):
        test = np.array([r and folds[g] == k for g, r in zip(group, rest)])
        oof[test] = model().fit(X[~test], y[~test]).predict_proba(X[test])[:, 1]

    neg = oof[~y]
    threshold = float(np.quantile(neg, 0.99))  # at most 1% of clean pictures above it
    print(f"held-out AUC {roc_auc_score(y, oof):.2f}; threshold for 1% false alarms: {threshold:.3f}")
    for g in pos_groups:
        sel = [i for i, gg in enumerate(group) if gg == g and y[i]]
        cam = train[sel[0]]["camera"]
        print(
            f"  {str(cam)[:22]:22} {len(sel):2} blurred, caught when unseen: {sum(oof[i] >= threshold for i in sel)}"
            f"  (scores {', '.join(f'{oof[i]:.2f}' for i in sel[:8])}{'…' if len(sel) > 8 else ''})"
        )

    final = model().fit(X, y)
    p = final.predict_proba(features(new, emb))[:, 1]
    order = np.argsort(-p)
    flagged = [new[i] for i in order if p[i] >= threshold]
    print(
        f"new dataset: {len(flagged)} of {len(new)} above the threshold; "
        f"already excluded by the rules: {sum(bool(r['excluded']) for r in flagged)}"
    )
    out = [{"id": new[i]["id"], "blur_model_score": round(float(p[i]), 4)} for i in order]
    Path("data/blur_model_new.json").write_text(json.dumps({"threshold": threshold, "scores": out}))
    print("top cameras:", Counter(str(r["camera"])[:16] for r in flagged).most_common(6))


if __name__ == "__main__":
    main()
