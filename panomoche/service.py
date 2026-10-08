"""Local web app and HTTP API.

    GET  /               upload page: score your own pictures live
    GET  /review         review page of a predictions file (``serve --predictions``)
    GET  /images/{id}    picture from that file, ``?size=thumb|full``
    GET  /api/info       scorer, model and thresholds
    POST /predict        multipart ``picture`` (+ optional ``is_pano``, ``sgblur``) -> prediction + semantics
    POST /api/sequence   compare already-scored pictures as one sequence, in the given order

``/predict`` is also what a Panoramax backend would call on upload, the way
GeoVisio delegates face and plate blurring to an external SGBlur service.
"""

from __future__ import annotations

import io
import threading
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image

from . import __version__
from .imaging import load_image, prepare
from .panoramax import DEFAULT_TAG_KEY, read_jsonl, semantics_for
from .privacy import blurred_share, parse_sgblur_header, privacy_from_file
from .report import render_review, script_json, template
from .sequence import sequence_summary
from .sharpness import measure

MAX_UPLOAD = 60 * 2**20  # an HD 360° picture is ~10-25 MB
THUMB_WIDTH = 480


@lru_cache(maxsize=2048)
def _thumbnail(path: str) -> bytes:
    img = Image.open(path).convert("RGB")
    img.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


PRIVACY_FIELDS = ("privacy_blur", "privacy_patches", "privacy_record", "privacy_boxes")
KNOWN_FIELDS = ("collection", "rank", "camera", "original_size", "gps_accuracy", "pixel_density", "datetime")


def _redecide(rows: list[dict], rules, decide) -> None:
    """Re-decide stored predictions with the given rules. Pictures judged with neighbours fetched from
    Panoramax (``context_fetched``, not in the file) keep that judgement."""
    for r in rows:
        if "rules_excluded" in r:  # files written by older versions, which had reviewer labels
            r["excluded"] = r.pop("rules_excluded")
    kept = {
        r["id"]: {k: r.get(k) for k in ("excluded", "issues", "low_quality", "context")}
        for r in rows
        if r.get("context_fetched")
    }
    if rows:
        decide(rows, rules)
    for r in rows:
        if r["id"] in kept:
            r.update(kept[r["id"]])


def create_app(
    score,
    info: dict,
    predictions: Path | None = None,
    rules=None,
    orientation=None,
    waviness: bool = False,
    ads: bool = False,
) -> FastAPI:
    """``score`` is a scorer from ``scoring.make_scorer`` (or any callable with the same contract).

    Every uploaded picture is fully measured (privacy blur, orientation, sharpness, TOPIQ), then
    sorted by the same decision as ``predict`` (``decision.py``): alone for a single picture, as
    one sequence for ``/api/sequence``. Privacy blur comes from SGBlur's record: the ``sgblur``
    form field (its ``x-sgblur`` response header, what a Panoramax backend has on upload) or the
    comment SGBlur leaves at the end of an original Panoramax file.

    ``waviness`` / ``ads``: also run the merged waviness and advertisement checks on uploads
    (``serve`` turns both on; the advertisement model loads on first use).
    """
    from .decision import Rules, decide

    rules = rules or Rules()
    info = {
        **info,
        "waviness_check": waviness,
        "ads_check": ads,
        "min_wavy_neighbours": rules.min_wavy_neighbours,
        "max_privacy_blur": rules.max_privacy_blur,
        "max_blurred_area": rules.max_blurred_area,
        "confirm_cpbd": rules.confirm_cpbd,
        "soft_blurred_area": rules.soft_blurred_area,
        "soft_cpbd": rules.soft_cpbd,
        "orientation_check": orientation is not None,
    }
    rows = read_jsonl(predictions) if predictions else []
    # The review page uses the same decision as uploads, with this server's rules: verdicts stored by an
    # earlier ``predict`` (maybe with other rules) are recomputed from the stored measurements.
    _redecide(rows, rules, decide)
    by_id = {r["id"]: r for r in rows}
    lock = threading.Lock()  # one picture at a time: bounded memory and CPU on small machines
    app = FastAPI(title="panomoche", version=__version__)

    @app.get("/", response_class=HTMLResponse)
    def upload_page():
        return template("upload.html").replace("__INFO__", script_json({**info, "predictions": len(rows)}))

    @app.get("/api/info")
    def api_info():
        return {**info, "predictions": len(rows)}

    @app.post("/predict")
    def predict(
        picture: UploadFile = File(...),
        is_pano: bool | None = Form(None),
        tag_key: str = Form(DEFAULT_TAG_KEY),
        sgblur: str | None = Form(None),
    ):
        data = picture.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(status_code=413, detail=f"picture larger than {MAX_UPLOAD // 2**20} MB")
        try:
            img = load_image(io.BytesIO(data))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"unreadable picture: {exc}") from exc
        start = time.perf_counter()
        prep = prepare(img, is_pano)
        if sgblur:
            try:
                raw_w, raw_h = Image.open(io.BytesIO(data)).size  # SGBlur boxes use the stored orientation
                privacy = blurred_share(parse_sgblur_header(sgblur), raw_w, raw_h, prep.is_pano)
            except (ValueError, TypeError, AttributeError) as exc:
                raise HTTPException(status_code=422, detail=f"unreadable sgblur field: {exc}") from exc
        else:
            privacy = privacy_from_file(data, prep.is_pano, original=False)
        known = by_id.get(Path(picture.filename or "").stem)  # a picture of the review page (named by its id)
        if known:
            if privacy.get("privacy_record") != "sgblur":
                privacy = {k: known[k] for k in PRIVACY_FIELDS if k in known} or privacy
            original = known.get("original_size")
        else:
            original = None
        row = {
            "id": picture.filename or "upload",
            "is_pano": prep.is_pano,
            "width": img.width,
            "height": img.height,
            **privacy,
            **measure(
                img, prep.is_pano, privacy.get("privacy_boxes"), original_width=original[0] if original else None
            ),
        }
        if known:  # what the review page knows beyond the file: metadata and the sequence
            row.update({k: known[k] for k in KNOWN_FIELDS if k in known}, review_id=known["id"])
        with lock:
            if orientation is not None:
                from .orientation import predict as predict_orientation

                row.update(predict_orientation(*orientation, prep, image=img))
            if waviness:
                from .waviness import check as check_waviness

                row.update(check_waviness(img, prep.is_pano))
            if ads:
                from .advertising import check as check_ads

                row.update(check_ads(img))
            row.update(score(prep))
        if known and known.get("collection") and known.get("rank") is not None:
            # Judged among its sequence neighbours, exactly as on the review page.
            seq = [dict(r) for r in rows if r.get("collection") == known["collection"] and r["id"] != known["id"]]
            decide([*seq, row], rules)
        else:
            decide([row], rules)
        row["elapsed_ms"] = round((time.perf_counter() - start) * 1000)
        row["semantics"] = semantics_for(row, tag_key)  # TOPIQ findings only; hard exclusions aren't tagged
        return row

    @app.post("/api/sequence")
    def compare_sequence(payload: dict = Body(...)):
        """Re-decide already measured pictures as one sequence, in the given order."""
        fields = (
            "is_pano",
            "quality_score",
            "blurred_area",
            "cpbd",
            "privacy_blur",
            "orientation",
            "orientation_confidence",
            "orientation_views",
            "issues",
            "gps_accuracy",
            "pixel_density",
            "waviness_label",
            "waviness_score",
            "ad_rejected",
        )
        seq = [
            {"id": str(p["id"]), "collection": "upload", "rank": i + 1, **{k: p.get(k) for k in fields}}
            for i, p in enumerate(payload.get("pictures", []))
        ]
        for r in seq:
            r["issues"] = list(r.get("issues") or [])
        decide(seq, rules)
        summary = sequence_summary(seq)
        return {
            "pictures": [{k: r.get(k) for k in ("id", "issues", "low_quality", "excluded", "context")} for r in seq],
            "summary": summary[0] if summary else None,
            "params": {"window": rules.window, "drop": rules.max_topiq_below_neighbours},
        }

    @app.get("/review", response_class=HTMLResponse)
    def review():
        if not rows:
            return HTMLResponse(
                "<p>No predictions loaded. Start the server with "
                "<code>panomoche serve --predictions data/sequences.jsonl</code>.</p>",
                status_code=404,
            )
        return render_review(
            rows,
            f"Review: {predictions.name}",
            lambda r, size: f"images/{quote(r['id'])}?size={size}",
            upload_url="./",
        )

    @app.get("/images/{picture_id}")
    def image(picture_id: str, size: str = "thumb"):
        r = by_id.get(picture_id)  # only pictures listed in the predictions file are served
        if r is None or not Path(r["path"]).is_file():
            raise HTTPException(status_code=404, detail="unknown picture")
        if size == "full":
            return FileResponse(r["path"], media_type="image/jpeg")
        return Response(_thumbnail(r["path"]), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    return app
