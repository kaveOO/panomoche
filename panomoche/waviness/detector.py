"""Explainable image waviness heuristic: Canny chains + shared nonlinear bending.

Detects rolling-shutter "wobble": long, mostly vertical edges (facades, poles)
bending together in a repeated way. Scores are heuristic indices, not
probabilities. Merged from the standalone Waviness Check v0.2: its original
documentation is in docs/merged-tools/waviness-check.md, how panomoche uses it in
docs/METHODS.md ("Waviness and advertisement checks").
"""
from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

VERSION = "0.2.0"
MAX_PIXELS = 40_000_000
# Not set globally (Image.MAX_IMAGE_PIXELS): it would also cap panomoche's reading of large 360° originals.
# load_image checks MAX_PIXELS before decoding instead.


@dataclass
class Settings:
    max_dimension: int = 1280
    minimum_span: float = 0.09
    minimum_bending_percent: float = 0.20
    correlation_threshold: float = 0.78
    decision_threshold: float = 45.0

    def validate(self):
        if not 640 <= self.max_dimension <= 1920:
            raise ValueError("Analysis resolution must be between 640 and 1920 pixels.")
        if not 0.05 <= self.minimum_span <= 0.3:
            raise ValueError("Minimum edge span must be between 0.05 and 0.30.")
        if not 0.05 <= self.minimum_bending_percent <= 2:
            raise ValueError("Minimum bending must be between 0.05% and 2%.")
        if not 0.5 <= self.correlation_threshold <= 0.99:
            raise ValueError("Correlation threshold must be between 0.50 and 0.99.")
        if not 0 <= self.decision_threshold <= 100:
            raise ValueError("Decision threshold must be between 0 and 100.")
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError("Settings must be finite numbers.")


def load_image(data: bytes) -> np.ndarray:
    """Decode once, honor orientation, and composite transparency onto white."""
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 40 megapixel limit.")
            if min(source.size) < 96:
                raise ValueError("Both image dimensions must be at least 96 pixels.")
            image = ImageOps.exif_transpose(source)
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
            return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Cannot read this image. Use a PNG, JPEG, WebP, BMP or TIFF.") from error


def detrend(values: np.ndarray, degree: int = 1) -> np.ndarray:
    t = np.linspace(-1, 1, len(values))
    return values - np.polynomial.polynomial.polyval(
        t, np.polynomial.polynomial.polyfit(t, values, degree)
    )


def smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(values.reshape(-1, 1), (1, 0), 0,
                            sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101).ravel()


def amplitude(values: np.ndarray) -> float:
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def extract_edges(gray: np.ndarray, settings: Settings):
    h, w = gray.shape
    filtered = cv2.GaussianBlur(gray, (5, 5), 0.9)
    # Gradient quantiles adapt to contrast. These are extraction thresholds,
    # never a sharpness score or blur classifier.
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    nonzero = magnitude[magnitude > 6]
    high = float(np.clip(np.percentile(nonzero, 65), 40, 100)) if nonzero.size else 70.0
    low = high * 0.35
    edges = cv2.Canny(filtered, low, high, L2gradient=True)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    raw = []
    minimum_span = max(30, int(h * settings.minimum_span))

    for contour in contours:
        points = contour[:, 0, :].astype(np.float64)
        if len(points) < minimum_span:
            continue
        # Split closed contours into long chains with a stable vertical
        # direction. findContours alone returns entire window perimeters.
        tangent = np.roll(points, -4, axis=0) - np.roll(points, 4, axis=0)
        vertical = np.abs(tangent[:, 1]) > 0.65 * np.linalg.norm(tangent, axis=1)
        direction = np.where(vertical, np.sign(tangent[:, 1]), 0)
        cuts = np.r_[0, np.flatnonzero(direction[1:] != direction[:-1]) + 1, len(points)]
        for begin, end in zip(cuts[:-1], cuts[1:]):
            if end - begin < minimum_span or direction[begin] == 0:
                continue
            chain = points[begin:end]
            y0, y1 = int(chain[:, 1].min()), int(chain[:, 1].max())
            if y1 - y0 < minimum_span:
                continue
            ys, inverse = np.unique(chain[:, 1].astype(int), return_inverse=True)
            counts = np.bincount(inverse)
            xs = np.bincount(inverse, weights=chain[:, 0]) / counts
            if len(ys) / (y1 - y0 + 1) < 0.9 or np.max(np.diff(ys)) > 6:
                continue
            full_y = np.arange(y0, y1 + 1)
            full_x = np.interp(full_y, ys, xs)
            if np.max(np.abs(np.diff(full_x))) > 5:
                continue
            smoothed = smooth(full_x, max(2.0, len(full_x) * 0.015))
            linear = detrend(smoothed, 1)
            # Remove a quadratic bow as a conservative defense against lens
            # distortion and naturally curved structures; this is not lens
            # calibration and can miss a wave that spans only a short arc.
            nonlinear = detrend(smoothed, 2)
            noise = float(np.median(np.abs(full_x - smoothed)))
            bend = amplitude(nonlinear)
            is_bent = (bend >= w * settings.minimum_bending_percent / 100
                       and amplitude(linear) >= w * 0.003
                       and bend >= 5 * max(noise, 0.35))
            raw.append({"ys": full_y, "xs": full_x, "smooth": smoothed,
                        "linear": linear, "nonlinear": nonlinear,
                        "y0": y0, "y1": y1, "x": float(np.mean(full_x)),
                        "span": y1 - y0, "amplitude": amplitude(linear),
                        "bending": bend, "noise": noise, "bent": bool(is_bent)})

    # Canny/contours can trace both sides of the very same boundary. Keep
    # the longest version once, rather than treating it as corroboration.
    candidates = []
    for candidate in sorted(raw, key=lambda item: -item["span"]):
        duplicate = False
        for existing in candidates:
            lo = max(candidate["y0"], existing["y0"])
            hi = min(candidate["y1"], existing["y1"])
            if hi - lo < 0.65 * min(candidate["span"], existing["span"]):
                continue
            y = np.arange(lo, hi + 1)
            separation = np.median(np.abs(np.interp(y, candidate["ys"], candidate["xs"])
                                         - np.interp(y, existing["ys"], existing["xs"])))
            if separation < max(4, w * 0.004):
                duplicate = True
                break
        if not duplicate:
            candidates.append(candidate)
        if len(candidates) >= 180:
            break
    return edges, candidates, {"canny_low": round(low, 1), "canny_high": round(high, 1)}


def shared_bending(candidates: list, h: int, w: int, settings: Settings):
    matches = []
    # Short arcs of unrelated objects often look alike after a polynomial
    # fit. Extraction may retain them, but confirmation needs a longer
    # baseline, separately from the adjustable amplitude threshold.
    bent_indices = [i for i, c in enumerate(candidates)
                    if c["bent"] and c["span"] >= h * 0.15]
    for pos, i in enumerate(bent_indices):
        a = candidates[i]
        for j in bent_indices[pos + 1:]:
            b = candidates[j]
            lo, hi = max(a["y0"], b["y0"]), min(a["y1"], b["y1"])
            if hi - lo < max(h * 0.10, 0.6 * min(a["span"], b["span"])):
                continue
            ys = np.arange(lo, hi + 1)
            ax = np.interp(ys, a["ys"], a["smooth"])
            bx = np.interp(ys, b["ys"], b["smooth"])
            # Compare separation at the same image rows, rather than the
            # mean x of traces with different start/end rows.
            if np.median(np.abs(ax - bx)) < w * 0.025:
                continue
            # Compare the actual bending shape after removing tilt. Removing
            # a quadratic here can force short unrelated arcs into similar
            # cubic residuals and exaggerate correlation.
            ar, br = detrend(ax, 1), detrend(bx, 1)
            if min(amplitude(ar), amplitude(br)) < max(2, w * 0.001):
                continue
            correlation = float(np.corrcoef(ar, br)[0, 1])
            # Rotation can move different parts of a frame in opposite
            # directions, so matching shape may have either polarity.
            if abs(correlation) >= settings.correlation_threshold:
                matches.append({"a": i, "b": j, "correlation": round(correlation, 3),
                                "y0": lo, "y1": hi})
    supported = sorted({m[key] for m in matches for key in ("a", "b")})
    # Count spatially separated support clusters, not every neighboring edge.
    centers = []
    for i in sorted(supported, key=lambda k: candidates[k]["x"]):
        if not centers or candidates[i]["x"] - centers[-1] >= w * 0.025:
            centers.append(candidates[i]["x"])
    return matches, supported, len(centers)


def analyze(image: np.ndarray, settings: Settings | None = None):
    settings = settings or Settings()
    settings.validate()
    original_h, original_w = image.shape[:2]
    factor = min(1.0, settings.max_dimension / max(original_h, original_w))
    resized = cv2.resize(image, (round(original_w * factor), round(original_h * factor)),
                         interpolation=cv2.INTER_AREA) if factor < 1 else image.copy()
    h, w = resized.shape[:2]
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges, candidates, extraction = extract_edges(gray, settings)
    matches, supported, spatial_support = shared_bending(candidates, h, w, settings)
    usable = len(candidates)
    bent = sum(c["bent"] for c in candidates)
    total_length = sum(c["span"] for c in candidates)
    shared_length = sum(candidates[i]["span"] for i in supported)
    shared_coverage = shared_length / max(total_length, 1)
    long_edges = sum(c["span"] >= h * 0.15 for c in candidates)
    confidence = "low"
    if usable >= 6 and spatial_support >= 3 and shared_coverage >= 0.25:
        confidence = "high"
    elif usable >= 3:
        confidence = "medium"
    if spatial_support >= 2:
        bend_percent = float(np.median([candidates[i]["bending"] for i in supported]) / w * 100)
        strength = min(1.0, bend_percent / (settings.minimum_bending_percent * 2.2))
        agreement = float(np.median([abs(m["correlation"]) for m in matches]))
        # Straight reference edges count as counterevidence. A small set of
        # high-correlation curves cannot saturate a whole-image score.
        coverage_factor = min(1.0, shared_coverage / 0.30)
        score = 100 * strength * min(1, spatial_support / 3) * agreement * coverage_factor
        if spatial_support < 3:
            score = min(30.0, score)
    else:
        bend_percent = 0.0
        score = 0.0
    score = round(float(score), 1)
    if usable < 3 or (long_edges < 6 and spatial_support < 3):
        label, title = "insufficient_evidence", "Insufficient evidence"
        explanation = "Too few long, consistently bending sections were found to assess geometric waviness."
        confidence = "low"
    elif score >= settings.decision_threshold and spatial_support >= 3:
        label, title = "wavy", "Waviness detected"
        explanation = (f"Bending agrees across {spatial_support} separated edge groups, "
                       f"covering {shared_coverage:.0%} of usable edge length.")
    elif score >= settings.decision_threshold * 0.65:
        label, title = "review", "Review suggested"
        explanation = "Some edges bend, but the amplitude or independent agreement is limited."
    else:
        label, title = "no_strong_evidence", "No strong waviness evidence"
        explanation = (f"Only {shared_coverage:.0%} of usable edge length shows confirmed shared bending; "
                       "the remaining edges do not corroborate waviness.")
        # Confidence refers to evidence availability, not correctness probability.
        confidence = "medium" if usable >= 6 else "low"

    overlay = resized.copy()
    for i, candidate in enumerate(candidates):
        color = (224, 179, 99)  # cyan: usable reference edge (BGR)
        if candidate["bent"]:
            color = (76, 184, 250)  # amber: bending without corroboration
        if i in supported:
            color = (118, 111, 250)  # coral: corroborated bending
        points = np.column_stack((candidate["smooth"], candidate["ys"])).astype(np.int32)
        cv2.polylines(overlay, [points], False, color, 2, cv2.LINE_AA)
        if i in supported:
            p = (int(points[len(points) // 2, 0]), int(points[len(points) // 2, 1]))
            cv2.circle(overlay, p, 9, (25, 27, 38), -1, cv2.LINE_AA)
            cv2.putText(overlay, str(i + 1), (p[0] - 5, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 255, 255), 1, cv2.LINE_AA)

    summaries = []
    for i, c in enumerate(candidates):
        step = max(1, len(c["ys"]) // 80)
        summaries.append({"id": i + 1, "x": round(c["x"], 1), "start_y": c["y0"],
                          "end_y": c["y1"], "span_pixels": c["span"],
                          "deviation_percent": round(c["amplitude"] / w * 100, 3),
                          "bending_percent": round(c["bending"] / w * 100, 3),
                          "bent": c["bent"], "supported": i in supported,
                          "profile": [[int(y), round(float(r), 2)] for y, r in
                                      zip(c["ys"][::step], c["linear"][::step])]})
    result = {"version": VERSION, "label": label, "title": title, "score": score,
              "confidence": confidence, "explanation": explanation,
              "score_is_probability": False, "usable_edges": usable, "bending_edges": bent,
              "supported_edges": len(supported), "independent_groups": spatial_support,
              "shared_edge_coverage_percent": round(shared_coverage * 100, 1),
              "long_edges": long_edges,
              "median_bending_percent": round(bend_percent, 3),
              "original_size": [original_w, original_h], "analysis_size": [w, h],
              "settings": asdict(settings), "extraction": extraction, "edges": summaries,
              "matches": [{**m, "a": m["a"] + 1, "b": m["b"] + 1} for m in matches],
              "limitations": "A single-image heuristic for long, mostly vertical structures. "
                             "Natural repeated curves can resemble distortion; weak or missing edges can hide it. "
                             "Thresholds require validation on representative camera images."}
    return result, resized, overlay, edges


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode analysis image.")
    return encoded.tobytes()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="panomoche waviness", description="Detect shared edge waviness in images.")
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--threshold", type=float, default=45, help="Decision score, 0–100")
    parser.add_argument("--min-bending", type=float, default=0.20, help="Nonlinear bending as %% of image width")
    args = parser.parse_args(argv)
    settings = Settings(decision_threshold=args.threshold, minimum_bending_percent=args.min_bending)
    try:
        settings.validate()
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for index, path in enumerate(args.images, 1):
        try:
            result, _, overlay, edges = analyze(load_image(path.read_bytes()), settings)
            result["filename"] = path.name
            stem = f"{index:03d}-{path.stem}"
            (args.output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            (args.output_dir / f"{stem}-overlay.png").write_bytes(encode_png(overlay))
            (args.output_dir / f"{stem}-edges.png").write_bytes(encode_png(edges))
            print(f"{path.name}: {result['title']} | score {result['score']}/100 | "
                  f"{result['confidence']} evidence confidence")
        except (OSError, ValueError, cv2.error) as error:
            failed = True
            print(f"{path}: {error}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
