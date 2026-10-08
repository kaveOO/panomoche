"""Train the orientation classifier and check it.

1. train on the training sample (``data/raw``), save ``models/orientation.joblib``;
2. synthetic: accuracy on rotated copies of held-out pictures;
3. real: predict on every scored picture, save ``data/orientation_real.json``.

    python scripts/orientation_eval.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from panomoche.backbone import DinoBackbone
from panomoche.imaging import load_image, prepare
from panomoche.orientation import ROTATIONS, embed, predict, rotate, save, train
from panomoche.panoramax import read_jsonl


def main() -> None:
    backbone = DinoBackbone(threads=12)
    rng = np.random.default_rng(0)

    t = time.time()
    raw = sorted(Path("data/raw").glob("*.jpg"))
    meta = {r["id"]: r for r in read_jsonl(Path("data/raw/meta.jsonl"))}
    preps = [prepare(load_image(p), meta.get(p.stem, {}).get("is_pano")) for p in raw]
    clf = train(backbone, preps)
    save(clf)
    print(f"trained on {len(preps)} pictures x 4 rotations in {time.time() - t:.0f}s")

    held = []  # evaluation sets, when present (any predictions file of pictures not in data/raw)
    for name in ("random", "random2", "paris", "new"):
        if Path(f"data/{name}.jsonl").is_file():
            held += read_jsonl(Path(f"data/{name}.jsonl"))
    if not held:
        print("no evaluation set in data/: model saved, evaluation skipped")
        return
    sample = [held[i] for i in rng.permutation(len(held))[:200]]
    correct, confident_wrong, total = 0, 0, 0
    per_rot = {d: [0, 0] for d in ROTATIONS}
    for r in sample:
        views = prepare(load_image(r["path"]), r["is_pano"]).views()
        view = views[rng.integers(len(views))]
        for k, deg in enumerate(ROTATIONS):
            p = clf.predict_proba(embed(backbone, [rotate(view, deg)]))[0]
            ok = int(p.argmax()) == k
            correct += ok
            total += 1
            per_rot[deg][0] += ok
            per_rot[deg][1] += 1
            confident_wrong += (not ok) and p.max() >= 0.9
    print(
        f"synthetic: {correct}/{total} rotations recognised ({100 * correct / total:.1f}%), "
        f"{confident_wrong} wrong with confidence >= 0.9"
    )
    print("  per rotation: " + ", ".join(f"{d}°: {a}/{n}" for d, (a, n) in per_rot.items()))

    seq = read_jsonl(Path("data/sequences.jsonl")) if Path("data/sequences.jsonl").is_file() else []
    out = []
    t = time.time()
    for r in held + seq:
        res = predict(backbone, clf, prepare(load_image(r["path"]), r["is_pano"]))
        out.append({k: r.get(k) for k in ("id", "path", "is_pano", "camera", "rank", "collection")} | res)
    Path("data/orientation_real.json").write_text(json.dumps(out))
    flagged = [o for o in out if o["orientation"] != 0 and o["orientation_confidence"] >= 0.9]
    print(
        f"real: {len(out)} pictures in {time.time() - t:.0f}s, {len(flagged)} predicted rotated with confidence >= 0.9"
    )
    for o in flagged:
        print(f"   {o['camera']} #{o.get('rank')}: rotated {o['orientation']}° ({o['orientation_confidence']:.2f})")


if __name__ == "__main__":
    main()
