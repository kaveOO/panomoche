"""Command line entry point: ``panomoche {fetch,predict,tag,serve}``, plus the merged tools
``waviness``, ``waviness-ui``, ``ads``, ``ads-ui`` and ``ads-setup``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from .panoramax import (
    DEFAULT_API,
    DEFAULT_TAG_KEY,
    PanoramaxClient,
    item_record,
    read_jsonl,
    semantics_for,
    write_jsonl,
)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Diverse default sampling areas (lon_min, lat_min, lon_max, lat_max): cities, villages, mountains, coast.
DEFAULT_BBOXES = [
    [2.30, 48.84, 2.38, 48.88],
    [4.80, 45.73, 4.88, 45.78],
    [-1.72, 48.09, -1.64, 48.13],
    [1.40, 43.58, 1.48, 43.63],
    [5.35, 43.27, 5.42, 43.32],
    [3.03, 50.61, 3.10, 50.65],
    [-1.59, 47.20, -1.52, 47.23],
    [-0.62, 44.82, -0.55, 44.86],
    [7.72, 48.56, 7.78, 48.60],
    [5.70, 45.15, 5.78, 45.20],
    [-3.00, 48.00, -2.80, 48.15],
    [2.90, 45.60, 3.10, 45.75],
    [6.80, 45.85, 6.95, 45.95],
    [-1.65, 49.60, -1.55, 49.66],
    [3.85, 43.59, 3.91, 43.63],
    [8.70, 41.90, 8.76, 41.94],
    [6.00, 48.60, 6.30, 48.80],
    [5.00, 47.30, 5.07, 47.34],
]


def _bbox(s: str) -> list[float]:
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("bbox must be lon_min,lat_min,lon_max,lat_max")
    return v


def _list_images(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p]
    return out


def _meta(folder: Path) -> dict[str, dict]:
    """Records written by ``fetch`` (collection, rank, is_pano, instance...), by picture id."""
    path = folder / "meta.jsonl"
    return {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}


def _privacy(a, session, path: Path | None, record: dict, is_pano: bool | None) -> dict:
    """How much of the picture Panoramax's anonymisation blurred (see ``privacy.py``)."""
    from .privacy import privacy_from_file, privacy_from_url

    if a.no_privacy_check:
        return {}
    if record.get("hd_url"):
        return privacy_from_url(session, record["hd_url"], is_pano, a.privacy_full_download)
    return privacy_from_file(path.read_bytes(), is_pano, original=False)


_ORIENTATION = {}  # loaded once: (backbone, classifier) or None


def _orientation_checker(a):
    """DINOv3 + rotation classifier, if a trained model exists and the check isn't disabled."""
    if "checker" not in _ORIENTATION:
        path = getattr(a, "orientation_model", None)
        if getattr(a, "no_orientation", False) or not path or not Path(path).exists():
            _ORIENTATION["checker"] = None
        else:
            from .backbone import DinoBackbone
            from .orientation import load

            _ORIENTATION["checker"] = (DinoBackbone(threads=a.threads), load(path))
    return _ORIENTATION["checker"]


def _rules(a):
    """Decision thresholds from the command line (see ``decision.py``)."""
    from .decision import Rules

    return Rules(
        max_privacy_blur=a.max_privacy_blur,
        min_orientation_confidence=getattr(a, "min_orientation_confidence", 0.99),
        max_gps_accuracy=getattr(a, "max_gps_accuracy", 0.0),
        min_pixel_density=getattr(a, "min_pixel_density", 15),
        min_wavy_neighbours=2.0 if getattr(a, "no_waviness", False) else getattr(a, "min_wavy_neighbours", 0.30),
        max_blurred_area=a.max_blurred_area,
        confirm_cpbd=getattr(a, "max_cpbd", 0.15),
        max_topiq_below_neighbours=getattr(a, "seq_drop", 0.12),
        topiq_floor_pano=getattr(a, "abs_threshold", 0.35),
        topiq_floor_flat_alone=getattr(a, "abs_threshold_flat", 0.40),
        window=getattr(a, "seq_window", 10),
    )


def _measure(a, row: dict, path: Path, prep) -> dict:
    """Every measurement the decision needs except privacy blur and TOPIQ:
    orientation, sharpness, waviness and advertisement."""
    from .imaging import load_image
    from .sharpness import measure

    img = load_image(path)
    checker = _orientation_checker(a)
    if checker:
        from .orientation import predict as orientation

        row.update(orientation(*checker, prep, image=img))
    original = row.get("original_size")
    row.update(measure(img, row["is_pano"], row.get("privacy_boxes"), original_width=original[0] if original else None))
    if not getattr(a, "no_waviness", False):
        from .waviness import check as waviness

        row.update(waviness(img, row["is_pano"]))
    if not getattr(a, "no_ads", False):
        from .advertising import check as advertising

        row.update(advertising(img))
    return row


def _privacy_report(rows: list[dict]) -> None:
    records = Counter(r.get("privacy_record") for r in rows if "privacy_record" in r)
    if records:
        print("  privacy blur records: " + ", ".join(f"{k} {n}" for k, n in records.most_common()))


def cmd_fetch(a):
    """Download a sample. Only checks that need no picture context run here (metadata, privacy
    blur); everything else is decided by ``predict``, which can compare pictures with their sequence."""
    from tqdm import tqdm

    from .dedupe import drop_duplicates, known_collections, known_ids
    from .metadata import metadata_exclusion
    from .privacy import is_excluded

    client = PanoramaxClient(a.api)
    skip = known_ids(a.skip_known)
    if a.random:
        items = client.random_sample(
            a.random,
            seed=a.seed,
            skip_ids=skip,
            skip_collections=known_collections(a.skip_known) if a.new_sequences else (),
        )
    else:
        items = client.sample(a.bbox or DEFAULT_BBOXES, a.per_bbox, a.max_per_collection, a.seed)
    items = [i for i in items if i["id"] not in skip]
    print(
        f"sampled {len(items)} pictures not judged before ({len(skip)} known ids skipped), "
        f"downloading '{a.asset}' assets to {a.out}",
        file=sys.stderr,
    )
    records, excluded = [], 0
    for item in tqdm(items, unit="pic"):
        try:
            record = item_record(item)
            record.update(_privacy(a, client.session, None, record, record["is_pano"]))
            if metadata_exclusion(record, a.max_gps_accuracy, a.min_pixel_density) or is_excluded(
                record, a.max_privacy_blur
            ):
                excluded += 1
                continue
            client.download(item, a.out, a.asset)
            records.append(record)
        except Exception as exc:
            tqdm.write(f"skip {item['id']}: {exc}")
    for r in records:
        r["path"] = str(a.out / f"{r['id']}.jpg")
    reference = [
        r
        for path in a.skip_known
        if Path(path).suffix == ".jsonl" and Path(path).is_file()
        for r in read_jsonl(Path(path))
        if r.get("path")
    ]
    records, dups = drop_duplicates(records, reference)
    for d in dups:
        Path(d["path"]).unlink(missing_ok=True)
    if dups:  # what was removed, and which picture each one repeats
        old_dups = read_jsonl(a.out / "duplicates.jsonl") if (a.out / "duplicates.jsonl").exists() else []
        write_jsonl(a.out / "duplicates.jsonl", [*old_dups, *dups])
    print(
        f"kept {len(records)}, excluded {excluded} for metadata or more than {a.max_privacy_blur:.0%} privacy blur, "
        f"removed {len(dups)} duplicates",
        file=sys.stderr,
    )
    _privacy_report(records)
    meta = a.out / "meta.jsonl"
    old = {r["id"]: r for r in read_jsonl(meta)} if meta.exists() else {}
    write_jsonl(meta, {**old, **{r["id"]: r for r in records}}.values())


def cmd_predict(a):
    from tqdm import tqdm

    from .decision import decide
    from .imaging import load_image, prepare
    from .scoring import make_scorer
    from .sequence import sequence_summary

    score, _ = make_scorer(a.abs_threshold, a.abs_threshold_flat, a.threads)
    jobs: list[tuple[Path, dict]] = []  # (local path, picture metadata)
    meta: dict[str, dict] = {}
    for p in a.paths:
        if p.is_dir():
            meta.update(_meta(p))
    jobs += [(p, meta.get(p.stem, {"id": p.stem})) for p in _list_images(a.paths)]
    client = PanoramaxClient(a.api)
    if a.bbox or a.collection:
        items = client.search(bbox=a.bbox, limit=a.limit) if a.bbox else []
        for collection in a.collection or []:
            items += list(client.collection_items(collection, a.limit, a.start_after_rank))
        for item in tqdm(items, desc="download", unit="pic"):
            try:
                path = client.download(item, a.cache, a.asset)
            except Exception as exc:
                tqdm.write(f"skip {item['id']}: {exc}")
                continue
            jobs.append(
                (
                    path,
                    {
                        **item_record(item),
                        "thumb_url": item["assets"].get("thumb", {}).get("href"),
                        "item_url": next((link["href"] for link in item["links"] if link["rel"] == "self"), None),
                    },
                )
            )

    rows = []
    for path, extra in tqdm(jobs, desc="predict", unit="pic"):
        try:
            prep = prepare(load_image(path), extra.get("is_pano"))
        except Exception as exc:
            tqdm.write(f"skip {path}: {exc}")
            continue
        row = {
            **extra,
            "path": str(path),
            "is_pano": prep.is_pano,
            **_privacy(a, client.session, path, extra, prep.is_pano),
        }
        rows.append({**_measure(a, row, path, prep), **score(prep)})
    if a.no_sequence:
        for r in rows:
            r["rank"] = None  # judge every picture alone
    labels = json.loads(a.labels.read_text()) if a.labels and a.labels.is_file() else {}
    for r in rows:
        r["reviewer_label"] = labels.get(r["id"])  # pictures marked Blurred on the review page are excluded
    decide(rows, _rules(a))  # the one decision: excluded or kept

    write_jsonl(a.out, rows)
    compared = sum(bool(r.get("context")) for r in rows)
    reasons = Counter(r["excluded"] for r in rows if r.get("excluded"))
    print(
        f"{len(rows)} pictures ({compared} judged against their sequence): "
        f"{len(rows) - sum(reasons.values())} kept, {sum(reasons.values())} excluded "
        f"({', '.join(f'{k} {n}' for k, n in reasons.most_common()) or 'none'}) -> {a.out}"
    )
    _privacy_report(rows)
    if summary := sequence_summary(rows):
        print("sequences, worst median first:")
        for q in summary:
            median = "  –  " if q["median"] is None else f"{q['median']:.2f}"
            print(
                f"  {q['collection']}  {q['pictures']:4d} pics  median {median}  "
                f"excluded {q['excluded']:3d}  {q['camera'] or ''}"
            )
    if a.html:
        from .report import html_report

        html_report(rows, a.html)
        print(f"review sheet: {a.html}")


def _report_comment(p: dict, issues: list[str]) -> str:
    detail = f"score {p['quality_score']:.2f}"
    if seq := p.get("sequence"):
        detail += f" vs sequence median {seq['median']:.2f}"
    return f"Automated detection ({', '.join(issues)}; {detail}) by {p['model']}"


def cmd_tag(a):
    preds = [p for p in read_jsonl(a.predictions) if p.get("collection") and p.get("instance_api")]
    token = a.token or os.environ.get("PANORAMAX_TOKEN")
    if a.apply and not token:
        sys.exit("--apply needs --token or PANORAMAX_TOKEN")
    client = PanoramaxClient(token=token)
    n = 0
    for p in preds:
        issues = p.get("issues") or []
        if not issues:
            continue
        tags = semantics_for({**p, "issues": issues}, a.tag_key)
        n += 1
        if a.apply:
            client.add_semantics(p["instance_api"], p["collection"], p["id"], tags)
            if a.report:
                client.report(p["instance_api"], p["id"], _report_comment(p, issues))
        else:
            out = {
                "PATCH": f"{p['instance_api']}/collections/{p['collection']}/items/{p['id']}",
                "body": {"semantics": tags},
            }
            if a.report:
                out["REPORT"] = _report_comment(p, issues)
            print(json.dumps(out))
    verb = "tagged" if a.apply else "would tag (dry run, pass --apply to write)"
    print(f"{verb} {n} pictures", file=sys.stderr)


def cmd_serve(a):
    import uvicorn

    from .scoring import make_scorer
    from .service import create_app

    score, info = make_scorer(a.abs_threshold, a.abs_threshold_flat, threads=a.threads)
    app = create_app(
        score,
        info,
        a.predictions,
        rules=_rules(a),
        orientation=_orientation_checker(a),
        labels=None if a.no_labels else a.labels,
        waviness=not a.no_waviness,
        ads=not a.no_ads,
    )
    print(
        f"open http://{a.host}:{a.port}/  (upload page)"
        + (f"  and http://{a.host}:{a.port}/review" if a.predictions else ""),
        file=sys.stderr,
    )
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def _content_args(p) -> None:
    g = p.add_argument_group("waviness and advertisement (merged tools, see docs/METHODS.md)")
    g.add_argument("--no-waviness", action="store_true", help="skip the rolling-shutter waviness check")
    g.add_argument(
        "--min-wavy-neighbours",
        type=float,
        default=0.30,
        help="exclude a wavy flat picture when at least this share of its neighbours is wavy too",
    )
    g.add_argument("--no-ads", action="store_true", help="skip the advertisement check (SigLIP 2 + OCR)")


def _orientation_args(p) -> None:
    from .orientation import DEFAULT_MIN_CONFIDENCE, DEFAULT_MODEL

    g = p.add_argument_group("orientation (upside-down or sideways pictures)")
    g.add_argument(
        "--orientation-model",
        type=Path,
        default=DEFAULT_MODEL,
        help="rotation classifier from scripts/orientation_eval.py; the check is skipped if it is missing",
    )
    g.add_argument(
        "--min-orientation-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="exclude a picture when it looks rotated with at least this probability",
    )
    g.add_argument("--no-orientation", action="store_true", help="skip the orientation check")


def _metadata_args(p) -> None:
    from .metadata import DEFAULT_MAX_GPS_ACCURACY, DEFAULT_MIN_PIXEL_DENSITY

    g = p.add_argument_group("Panoramax metadata (a missing value never excludes)")
    g.add_argument(
        "--max-gps-accuracy",
        type=float,
        default=DEFAULT_MAX_GPS_ACCURACY,
        help="exclude pictures whose GPS accuracy is worse than this, in metres (default 0: disabled)",
    )
    g.add_argument(
        "--min-pixel-density",
        type=float,
        default=DEFAULT_MIN_PIXEL_DENSITY,
        help="exclude pictures below this many pixels per degree (0 disables; 360° GoPro Max = 16)",
    )


def _privacy_args(p, remote: bool = True) -> None:
    from .privacy import DEFAULT_MAX_PRIVACY_BLUR
    from .sharpness import DEFAULT_CONFIRM_CPBD, DEFAULT_MAX_BLURRED_AREA

    p.add_argument(
        "--max-blurred-area",
        type=float,
        default=DEFAULT_MAX_BLURRED_AREA,
        help="exclude pictures whose textured area is blurred at this share or more "
        "(1 disables; default %(default)s, see sharpness.py)",
    )
    p.add_argument(
        "--max-cpbd",
        type=float,
        default=DEFAULT_CONFIRM_CPBD,
        help="...and whose CPBD (share of edges that look sharp) is below this "
        "(default %(default)s; 1 lets the blurred area decide alone)",
    )
    g = p.add_argument_group("privacy blur (Panoramax face/plate anonymisation)")
    g.add_argument(
        "--max-privacy-blur",
        type=float,
        default=DEFAULT_MAX_PRIVACY_BLUR,
        help="exclude pictures where blurred faces/plates cover more than this share of the scored "
        "area, read from SGBlur's record in the original file (default %(default)s)",
    )
    if remote:
        g.add_argument("--no-privacy-check", action="store_true", help="don't read SGBlur records")
        g.add_argument(
            "--privacy-full-download",
            action="store_true",
            help="on instances without HTTP range support, download whole originals to read the record",
        )


# The merged standalone tools keep their own command lines; their arguments are passed through.
TOOLS = {
    "waviness": ("panomoche.waviness.detector", "batch waviness check: JSON report, overlay and edge map per picture"),
    "waviness-ui": ("panomoche.waviness.server", "browser UI of the waviness check (port 8765)"),
    "ads": ("panomoche.advertising.classifier", "batch advertisement check: JSON report per picture"),
    "ads-ui": ("panomoche.advertising.server", "browser UI of the advertisement check (port 8766)"),
    "ads-setup": ("panomoche.advertising.setup_models", "download and verify the SigLIP 2 model (~393 MiB)"),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in TOOLS:
        import importlib

        return importlib.import_module(TOOLS[argv[0]][0]).main(argv[1:])
    ap = argparse.ArgumentParser(prog="panomoche", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, (_, help_text) in TOOLS.items():
        sub.add_parser(name, help=help_text, add_help=False)

    p = sub.add_parser("fetch", help="download a diverse sample of Panoramax pictures")
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--bbox",
        type=_bbox,
        action="append",
        help="repeatable, use --bbox=LON_MIN,... for negative values; defaults to 18 areas across France",
    )
    p.add_argument(
        "--random",
        type=int,
        metavar="N",
        help="N pictures from random places across France, one per sequence (instead of --bbox)",
    )
    p.add_argument("--per-bbox", type=int, default=30)
    p.add_argument("--max-per-collection", type=int, default=3)
    p.add_argument("--asset", default="sd", choices=["sd", "hd", "thumb"])
    p.add_argument("--api", default=DEFAULT_API)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip-known",
        type=Path,
        nargs="*",
        default=[Path("data/all.jsonl"), Path("data/user_labels.json")],
        help="never fetch pictures listed in these predictions/labels files; also the reference "
        "for duplicate removal (default: %(default)s)",
    )
    p.add_argument(
        "--new-sequences",
        action="store_true",
        help="with --random: only sample sequences not in the --skip-known files",
    )
    _privacy_args(p)
    _metadata_args(p)
    p.set_defaults(fn=cmd_fetch, orientation_model=None, min_orientation_confidence=1.0)

    p = sub.add_parser("predict", help="score local pictures and/or Panoramax pictures")
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("data/user_labels.json"),
        help="reviewer labels: pictures marked Blurred are excluded (default %(default)s)",
    )
    p.add_argument("paths", type=Path, nargs="*")
    p.add_argument(
        "--abs-threshold",
        type=float,
        default=0.32,
        help="TOPIQ score under which a 360° picture, or any picture with neighbours, is low quality",
    )
    p.add_argument(
        "--abs-threshold-flat",
        type=float,
        default=0.40,
        help="the same for a flat picture judged alone, which TOPIQ rates higher",
    )
    p.add_argument("--bbox", type=_bbox)
    p.add_argument("--collection", action="append", help="sequence UUID, repeatable")
    p.add_argument(
        "--start-after-rank",
        type=int,
        help="with --collection: skip the first N pictures (instance API only, see --api)",
    )
    p.add_argument("--limit", type=int, default=100, help="pictures per --bbox or per --collection")
    p.add_argument("--api", default=DEFAULT_API)
    p.add_argument("--asset", default="sd", choices=["sd", "hd"])
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    g = p.add_argument_group("sequence context (see decision.py)")
    g.add_argument("--seq-window", type=int, default=10, help="neighbours considered on each side")
    g.add_argument(
        "--seq-drop",
        type=float,
        default=0.12,
        help="how far below its neighbours' median TOPIQ a picture may be before it is excluded",
    )
    g.add_argument("--no-sequence", action="store_true", help="judge every picture alone")
    _privacy_args(p)
    _metadata_args(p)
    _orientation_args(p)
    _content_args(p)
    p.add_argument("--out", type=Path, default=Path("predictions.jsonl"))
    p.add_argument("--html", type=Path, help="write a review contact sheet")
    p.add_argument("--threads", type=int)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("tag", help="write predictions back to Panoramax as semantics tags (dry run by default)")
    p.add_argument("predictions", type=Path)
    p.add_argument("--tag-key", default=DEFAULT_TAG_KEY)
    p.add_argument("--apply", action="store_true", help="actually send the PATCH requests")
    p.add_argument("--report", action="store_true", help="also open a picture_low_quality moderation report")
    p.add_argument("--token", help="Panoramax API token (or PANORAMAX_TOKEN)")
    p.set_defaults(fn=cmd_tag)

    p = sub.add_parser("serve", help="local web app (upload + review pages) and HTTP API")
    p.add_argument("--abs-threshold", type=float, default=0.32, help="topiq floor for 360° pictures")
    p.add_argument("--abs-threshold-flat", type=float, default=0.40, help="topiq floor for flat pictures alone")
    p.add_argument("--predictions", type=Path, help="predictions JSONL to browse at /review")
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("data/user_labels.json"),
        help="where the review page's Blurred / OK buttons save judgements (default %(default)s)",
    )
    p.add_argument("--no-labels", action="store_true", help="hide the labelling buttons")
    p.add_argument("--threads", type=int)
    _privacy_args(p, remote=False)
    _orientation_args(p)
    _content_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    if a.cmd == "predict" and not (a.paths or a.bbox or a.collection):
        ap.error("predict needs image paths, --bbox or --collection")
    a.fn(a)


if __name__ == "__main__":
    main()
