"""Local advertisement check; ambiguous content passes by default.

Merged from the standalone Advertisement Check v0.4: its original documentation is in
docs/merged-tools/advertisement-check.md, how panomoche uses it in docs/METHODS.md. Model data lives in ``models/siglip2`` (``panomoche ads-setup``
downloads it) or ``CONTENT_MODEL_DIR``.


SigLIP 2 similarities are evidence scores, not calibrated probabilities.
This module does not call or modify the waviness detector.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
VERSION = "0.4.0"
DEFAULT_THRESHOLD = 90.0
MODEL_REVISION = "ba1f3b0843f24bc5417d38e19c37b287d719b2f4"
MAX_PIXELS = 40_000_000
# Not set globally (Image.MAX_IMAGE_PIXELS): it would also cap panomoche's reading of large 360° originals.
# load_image checks MAX_PIXELS before decoding instead.


DEFAULT_MODEL_DIR = ROOT.parents[1] / "models" / "siglip2"


def model_directory():
    if os.environ.get("CONTENT_MODEL_DIR"):
        return Path(os.environ["CONTENT_MODEL_DIR"]).resolve()
    return DEFAULT_MODEL_DIR


def read_policy():
    data = (ROOT / "policy.json").read_bytes()
    policy = json.loads(data)
    entries = [(kind, category) for kind in ("reference", "advertising") for category in policy[kind]]
    prompts = [prompt.lower() for _, category in entries for prompt in category["prompts"]]
    return policy, entries, prompts, hashlib.sha256(data).hexdigest()


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def normalize(features):
    return features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)


def prepare_text_embeddings(directory):
    _, _, prompts, policy_hash = read_policy()
    cache = directory / "text_embeddings.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as stored:
            if str(stored["policy_hash"]) == policy_hash and str(stored["revision"]) == MODEL_REVISION:
                print("Text embeddings are ready.", flush=True)
                return
    print("Preparing policy text embeddings…", flush=True)
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=64)
    tokenizer.enable_padding(length=64, pad_id=1, pad_token="<pad>")
    encoder = session(directory / "text_model_quantized.onnx")
    vectors = []
    for start in range(0, len(prompts), 6):
        tokenized = tokenizer.encode_batch(prompts[start:start + 6])
        ids = np.array([row.ids for row in tokenized], dtype=np.int64)
        inputs = {"input_ids": ids}
        names = {item.name for item in encoder.get_inputs()}
        if "attention_mask" in names:
            inputs["attention_mask"] = np.array([row.attention_mask for row in tokenized], dtype=np.int64)
        vectors.append(encoder.run(["pooler_output"], inputs)[0])
    features = normalize(np.concatenate(vectors).astype(np.float32))
    np.savez(cache, features=features, policy_hash=policy_hash, revision=MODEL_REVISION)
    print(f"Prepared {len(prompts)} policy descriptions.", flush=True)


def load_image(data):
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Image exceeds 40 megapixels.")
            if min(source.size) < 96:
                raise ValueError("Both image dimensions must be at least 96 pixels.")
            image = ImageOps.exif_transpose(source)
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
            return image.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Cannot decode this image. Use PNG, JPEG, WebP, BMP or TIFF.") from error


# Signals describe commercial intent, not a required list of magic words.
COMMERCIAL_PATTERNS = {
    "call_to_action": r"\b(?:book|reserve|buy|shop|order|hire)\s+(?:now|today|online)\b|\b(?:call|contact)\s+us\b|\bbring\s+this\s+flyer\b",
    "offer": r"\b(?:happy\s+hour|deals?|special\s+offers?|limited\s+(?:time|offer)|sale|discount|promotions?|soldes|oferta)\b",
    "discount": r"\b\d+\s*%\s*(?:off|discount)|\b(?:half|[½]|1/2)\s*price\b|\bfree\s+shipping\b|\b(?:first|drink.s?)\b[^.!?]{0,25}\bfree\b",
    "price": r"[$€£]\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*(?:usd|eur|gbp)\b",
    "package": r"\b\d+\s*(?:days?|nights?)\b|\b(?:half\s+board|accommodation|accomodation)\b",
    "contact": r"\+\d[\d ()-]{6,}\d|\bwww\.[a-z0-9.-]+|\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"
}


def commercial_evidence(text):
    text = text.lower()
    signals = {key: sorted({match.group(0).strip() for match in re.finditer(pattern, text)})[:12]
               for key, pattern in COMMERCIAL_PATTERNS.items()}
    # Multiple prices strengthen commerce evidence. Contact details alone
    # carry little weight because ordinary documents and signs contain them.
    strength = min(1.0, 0.55 * bool(signals["call_to_action"]) + 0.20 * bool(signals["offer"])
                   + 0.40 * bool(signals["discount"]) + 0.60 * min(1, len(signals["price"]) / 3)
                   + 0.30 * bool(signals["package"]) + 0.15 * bool(signals["contact"]))
    return {"signals": signals, "strength": strength}


def empty_ocr(status):
    return {"available": False, "status": status, "words": 0, "text_area_percent": 0.0,
            "promotional_terms": [], "commercial_signals": {}, "commercial_strength": 0.0}


def ocr_evidence(image):
    command = shutil.which("tesseract")
    if not command:
        return empty_ocr("unavailable")
    working = image.copy()
    working.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    w, h = working.size
    with tempfile.TemporaryDirectory(prefix="content-ocr-") as directory:
        path = Path(directory) / "image.png"
        working.save(path)
        try:
            response = subprocess.run([command, str(path), "stdout", "-l", "eng", "--psm", "11", "tsv"],
                capture_output=True, text=True, timeout=15, check=True,
                env={**os.environ, "OMP_THREAD_LIMIT": "2"})
        except (subprocess.SubprocessError, OSError):
            return empty_ocr("failed")
    words, area = [], 0
    # Tesseract TSV has no quoting: a stray " read from the picture must not swallow the following words.
    for row in csv.DictReader(io.StringIO(response.stdout), delimiter="\t", quoting=csv.QUOTE_NONE):
        try:
            text = row["text"].strip()
            if text and float(row["conf"]) >= 45:
                words.append(text)
                area += int(row["width"]) * int(row["height"])
        except (ValueError, KeyError, TypeError):
            continue
    commerce = commercial_evidence(" ".join(words))
    return {"available": True, "status": "ready", "words": len(words),
            "text_area_percent": round(min(1, area / (w * h)) * 100, 2),
            "promotional_terms": sorted({term for values in commerce["signals"].values() for term in values}),
            "commercial_signals": commerce["signals"], "commercial_strength": round(commerce["strength"], 4)}


def score_evidence(reference_similarity, advertising_similarity, ocr=None):
    """Continuous evidence index; no term count or absolute similarity cutoff.

    The combination is a heuristic index, not a calibrated probability.
    Missing OCR leaves the visual score intact and marks analysis incomplete.
    """
    ocr = ocr or empty_ocr("disabled")
    values = (reference_similarity, advertising_similarity, ocr.get("text_area_percent", 0),
              ocr.get("commercial_strength", 0))
    if not all(math.isfinite(value) for value in values):
        return {"score": None, "complete": False, "visual_layout": None, "text_layout": None,
                "commercial_text": None, "clear_advertising_layout": False,
                "substantial_text": False, "commercial_intent": False, "strong_visual_poster": False}
    margin = advertising_similarity - reference_similarity
    visual = 1 / (1 + math.exp(-float(np.clip(margin / 0.012, -40, 40))))
    available = bool(ocr.get("available"))
    text = min(1.0, max(0.0, ocr.get("text_area_percent", 0)) / 8) if available else 0.0
    commerce = float(np.clip(ocr.get("commercial_strength", 0), 0, 1)) if available else 0.0
    # Independent contributions can strengthen evidence; absence of a
    # recognized word cannot erase the model's visual evidence.
    score = 100 * (1 - (1 - visual) * (1 - 0.55 * text) * (1 - 0.60 * commerce))
    return {"score": score, "complete": available,
            "visual_layout": 100 * visual, "text_layout": 100 * text if available else None,
            "commercial_text": 100 * commerce if available else None,
            "clear_advertising_layout": margin >= 0.005,
            "substantial_text": available and ocr.get("words", 0) >= 4 and ocr.get("text_area_percent", 0) >= 3,
            "commercial_intent": available and commerce >= 0.45,
            "strong_visual_poster": available and margin >= 0.04 and ocr.get("text_area_percent", 0) >= 8}


def decide(reference_similarity, advertising_similarity, threshold=DEFAULT_THRESHOLD, ocr=None):
    if not math.isfinite(threshold) or not 90 <= threshold <= 99:
        raise ValueError("Advertisement rejection threshold must be between 90 and 99.")
    evidence = score_evidence(reference_similarity, advertising_similarity, ocr)
    score = evidence["score"]
    if not evidence["complete"] or score is None:
        return "accept", score, "inconclusive_evidence"
    corroborated = evidence["clear_advertising_layout"] and evidence["substantial_text"] and (
        evidence["commercial_intent"] or evidence["strong_visual_poster"])
    if corroborated and score >= threshold:
        return "reject", score, "obvious_advertisement"
    return "accept", score, "no_obvious_advertisement"


class ContentFilter:
    def __init__(self, directory=None):
        self.directory = Path(directory or model_directory())
        vision = self.directory / "vision_model_quantized.onnx"
        if not vision.is_file():
            raise RuntimeError("Model data is missing. Run setup_models.py first.")
        prepare_text_embeddings(self.directory)
        self.policy, self.entries, self.prompts, self.policy_hash = read_policy()
        with np.load(self.directory / "text_embeddings.npz", allow_pickle=False) as data:
            self.text_features = data["features"].copy()
        if self.text_features.shape != (len(self.prompts), 768):
            raise RuntimeError("Policy text embeddings have an invalid shape. Run setup_models.py again.")
        if not np.isfinite(self.text_features).all():
            raise RuntimeError("Policy text embeddings contain invalid values. Run setup_models.py again.")
        self.encoder = session(vision)
        self.groups = []
        offset = 0
        for kind, category in self.entries:
            count = len(category["prompts"])
            self.groups.append((kind, category, slice(offset, offset + count)))
            offset += count

    def analyze(self, image, threshold=DEFAULT_THRESHOLD, use_ocr=True):
        started = time.monotonic()
        decide(0.2, 0.1, float(threshold))
        # One full-image view: no panorama sections or place-context vetoes.
        pixels = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
        pixels = np.transpose(pixels / 127.5 - 1, (2, 0, 1))[None, ...]
        try:
            vectors = normalize(self.encoder.run(["pooler_output"], {"pixel_values": pixels})[0])
            if not np.isfinite(vectors).all():
                raise RuntimeError("The visual model returned invalid evidence.")
            similarities = (vectors @ self.text_features.T)[0]
        except Exception as error:
            return unchecked_result(None, error, image.size, threshold)
        scores = np.array([np.mean(np.sort(similarities[span])[-2:]) for _, _, span in self.groups])
        reference_mask = np.array([kind == "reference" for kind, _, _ in self.groups])
        reference_index = int(np.argmax(np.where(reference_mask, scores, -10)))
        advertising_index = int(np.argmax(np.where(~reference_mask, scores, -10)))
        reference_score = float(scores[reference_index])
        advertising_score = float(scores[advertising_index])
        # Always measure text: absolute cosine similarity is not a portable
        # confidence cutoff, and natural advertising need not say BUY NOW.
        ocr = ocr_evidence(image) if use_ocr else empty_ocr("disabled")
        evidence = score_evidence(reference_score, advertising_score, ocr)
        decision, score, reason_code = decide(reference_score, advertising_score, float(threshold), ocr)
        reasons = {
            "obvious_advertisement": "A strong advertising layout and supporting text evidence identify a clear advertisement.",
            "no_obvious_advertisement": "No clear advertisement detected. Passed by default.",
            "inconclusive_evidence": "Advertising evidence is incomplete or uncertain. Passed by default."
        }
        ranking = [{"id": category["id"], "label": category["label"],
                    "advertising": kind == "advertising", "similarity": round(float(scores[index]), 4)}
                   for index, (kind, category, _) in enumerate(self.groups)]
        ranking.sort(key=lambda entry: -entry["similarity"])
        return {"version": VERSION, "scope": "obvious_advertisements_only",
                "decision": decision, "accepted": decision == "accept", "ad_rejected": decision == "reject",
                "checked": reason_code != "inconclusive_evidence",
                "detection_status": "partial" if reason_code == "inconclusive_evidence" else "complete",
                "ad_score": round(score, 1) if score is not None else None, "score_is_probability": False,
                "threshold": float(threshold), "reason_code": reason_code, "reason": reasons[reason_code],
                "best_reference_category": self.groups[reference_index][1]["id"],
                "advertising_similarity": round(advertising_score, 4),
                "reference_similarity": round(reference_score, 4),
                "advertising_margin": round(advertising_score - reference_score, 4),
                "score_components": {key: round(evidence[key], 1) if evidence[key] is not None else None
                                     for key in ("visual_layout", "text_layout", "commercial_text")},
                "evidence": {key: evidence[key] for key in ("clear_advertising_layout", "substantial_text",
                             "commercial_intent", "strong_visual_poster")},
                "categories": ranking, "ocr": ocr, "original_size": list(image.size),
                "seconds": round(time.monotonic() - started, 3),
                "model": "google/siglip2-base-patch16-224", "onnx_revision": MODEL_REVISION,
                "policy_version": self.policy["version"], "policy_sha256": self.policy_hash,
                "ambiguous_action": "accept", "waviness_checked": False,
                "limitations": "Conservative advertisement heuristic; not a map-relevance or quality check. "
                               "Can miss advertising. Does not verify GPS, provenance or whether a scene is synthetic."}


def unchecked_result(filename, error, size=None, threshold=DEFAULT_THRESHOLD):
    """An unavailable check is not evidence that a valid image is advertising."""
    return {"version": VERSION, "scope": "obvious_advertisements_only", "filename": filename,
            "decision": "accept", "accepted": True, "ad_rejected": False, "checked": False,
            "detection_status": "unavailable", "ad_score": None, "threshold": float(threshold),
            "reason_code": "check_unavailable", "reason": "Advertisement check unavailable. Passed by default.",
            "diagnostic": str(error), "original_size": list(size) if size else None,
            "ambiguous_action": "accept", "waviness_checked": False}


def failure_result(filename, error):
    # Input/transport errors are not content rejections or a review state.
    return {"version": VERSION, "filename": filename, "decision": "error", "accepted": False,
            "ad_rejected": False, "checked": False, "detection_status": "error",
            "reason_code": "processing_error", "reason": str(error), "ambiguous_action": "accept"}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="panomoche ads", description="Reject only obvious advertising graphics; other content passes.")
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("content-results"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Advertisement rejection threshold, 90–99. Higher values reject fewer images.")
    args = parser.parse_args(argv)
    try:
        decide(0.3, 0.1, args.threshold)
    except ValueError as error:
        parser.error(str(error))
    try:
        engine = ContentFilter()
    except Exception as error:
        engine = None
        model_error = str(error)
        print(f"Advertisement check unavailable: {error}. Valid images pass by default.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed = engine is None
    for index, path in enumerate(args.images, 1):
        try:
            image = load_image(path.read_bytes())
            result = engine.analyze(image, args.threshold) if engine else unchecked_result(path.name, model_error, image.size, args.threshold)
            result["filename"] = path.name
            failed |= result.get("detection_status") == "unavailable"
        except Exception as error:
            failed = True
            result = failure_result(path.name, error)
        (args.output_dir / f"{index:03d}-{path.stem}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print(f"{path.name}: {result['decision'].upper()} — {result['reason']}", flush=True)
    raise SystemExit(2 if failed else 0)


if __name__ == "__main__":
    main()
