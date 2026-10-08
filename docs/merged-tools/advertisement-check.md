# Advertisement Check

A local **advertisement-only check** for map imagery. Version 0.4 passes content by default and rejects only clear advertising graphics supported by visual layout and text evidence. There is no review queue. The waviness detector is unchanged and is never called.

Open **http://127.0.0.1:8766/**. Upload images or try the reference set, then export individual or batch JSON reports. Analysis stays on this computer and runs offline after model setup.

Landscapes, statues, animals, portraits, interiors, screenshots, documents, logos and ambiguous content are no longer rejected for their category. A shop sign, billboard, text or logo in a real photo is not sufficient for rejection. Passing means **no obvious advertisement was detected**; other Panoramax requirements and image quality belong to separate checks.

The default advertising-evidence threshold is **90/100**. Higher values require stronger evidence and reject fewer images. This is a heuristic index, not a probability, and the other corroborating checks must also pass.

## Run locally

```sh
./start.sh --port 8766
```

The launcher finds the prepared environment under `work/content-venv` and the cached model under `work/content-models`. Override these using `CONTENT_PYTHON` and `CONTENT_MODEL_DIR`. Stop a foreground server with Ctrl+C.

## Install elsewhere

Requires Python 3.10+. Tesseract with English language data enables OCR. If OCR or the visual model is unavailable, valid images pass unchecked instead of being rejected as advertising.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup_models.py
.venv/bin/python server.py --port 8766
```

On Windows use `.venv\Scripts\python.exe`. Setup downloads approximately 393 MiB from Hugging Face, verifies pinned SHA-256 hashes and caches text embeddings. Model binaries are excluded from the ZIP.

## Python and CLI

```python
from pathlib import Path
from classifier import ContentFilter, load_image

engine = ContentFilter()  # Initialize once and reuse.
report = engine.analyze(load_image(Path("photo.jpg").read_bytes()), threshold=90)
if report["ad_rejected"]:
    print("Reject obvious advertising:", report["reason"])
else:
    print("Continue with the separate quality and location checks.")
```

`checked: false` identifies incomplete or unavailable analysis. The example assumes a decodable image; normal input validation still applies. This tool is not connected to a live Panoramax instance and does not publish or delete images.

```sh
python classifier.py photos/*.jpg --output-dir content-results
python classifier.py photo1.png photo2.jpg --threshold 97 --output-dir fewer-rejections
```

Each input receives a numbered JSON report. Invalid files get `error` reports; processing continues. Exit 0 means the batch completed, not that all inputs passed. Exit 2 indicates a technical failure; valid images may still have passed unchecked. Read `ad_rejected` and `checked`. Use a fresh output directory to preserve earlier runs.

## HTTP API

POST `/api/analyze` with JSON:

```json
{"image":"data:image/jpeg;base64,...","filename":"photo.jpg","ad_threshold":90}
```

HTTP 200 returns `decision: "accept"` or `"reject"`, `accepted`, `ad_rejected`, `checked`, `ad_score`, evidence and a reason. Only `ad_rejected: true` identifies a content rejection. Model unavailability returns a default pass with `checked: false`. Malformed requests, undecodable images and transport failures are separate errors (`decision: "error"`, `ad_rejected: false`). Retry busy responses (HTTP 429).

**Migration:** the old `threshold` request field represented minimum place relevance and is now ignored. Use `ad_threshold` for the new advertising threshold. The old `relevance_score`, place-versus-off-topic scores and panorama admission rules have been removed.

The server binds to loopback and accepts image bytes rather than remote URLs. Inference is serialized; the UI processes images sequentially. Input limits remain 20 MiB per UI file, 32 MiB per request, 40 megapixels and a minimum side of 96 pixels.

## Decision method

1. Decode, apply EXIF orientation and composite transparency onto white.
2. Run one whole-image SigLIP 2 embedding. Compare against 64 descriptions in policy version 2.1, including restaurant, travel and retail advertising and non-advertising reference categories. Each category averages its two strongest description matches.
3. Always run OCR when enabled. Absolute cosine similarity is not a portable confidence cutoff; a small similarity cannot skip text checks or force a zero score.
4. Compute continuous components for advertising layout versus references, text coverage, and commercial wording. Commerce includes prices, offers, booking/contact prompts and accommodation packages. No particular phrase or number of matching keywords is required.
5. Combine the components into an evidence index. Reject only above the threshold **and** with an advertising-versus-reference margin of at least 0.005, at least four OCR words, text covering at least 3% of the image, and either substantial commercial intent or exceptionally strong poster evidence (margin at least 0.04 and text covering at least 8%). These gates preserve conservative acceptance.
6. Uncertain evidence passes. Missing OCR retains the visual score and marks analysis partial; unavailable model evidence is unknown (`null`), never a fabricated zero. No other content category rejects.

The formula is `100 × [1 − (1 − V) × (1 − 0.55 × T) × (1 − 0.60 × C)]`, where `V = sigmoid((advertising_similarity − reference_similarity) / 0.012)`, `T = min(text_area_percent / 8, 1)` and `C` is the commercial-text strength from 0 to 1. These weights define an **uncalibrated heuristic index**, not a probability. A score at or above the threshold does not override the corroboration gates. Decisions use full precision.

Commercial-text strength combines call-to-action evidence (0.55), offers (0.20), discounts (0.40), distinct prices (up to 0.60), package details (0.30) and contact details (0.15), capped at 1. Contact details alone are weak evidence. Text coverage alone cannot reject an ordinary document or photograph.

Reports include `score_components` (visual layout, text coverage and commercial wording), policy hash/version, model revision, raw similarities, OCR status, commercial signals, decision conditions, timing and threshold. The UI shows the components separately so a zero commercial-wording component cannot be confused with zero visual evidence.

The pretrained model is unchanged: no training or fine-tuning is performed on our fixtures. Description embeddings refresh when `policy.json` changes. Our earlier tests relied too heavily on controlled advertising graphics; two supplied real restaurant/resort flyers now cover that failure. They are regression examples used while developing this revision, not held-out evaluation data.

## Verification and limits

```sh
python -m unittest -v test_filter
```

The suite covers default acceptance, advertising corroboration, threshold direction, missing OCR/model behavior, technical errors, EXIF orientation, HTTP uploads, migration and exports. All 19 examples are checked: 12 non-advertising images pass and 7 advertising graphics reject at the default threshold, including the two supplied real advertisements. Crops of the food and scenery without advertising overlays also pass. The supplied statue and ice landscape also pass at reduced resolution.

These examples are development regressions, not an accuracy benchmark. The filter favors avoiding false rejections and can miss advertisements, particularly subtle designs, stylized text or other languages. It does not reject other unrelated content or validate GPS, permissions, provenance, authenticity or quality.

Verified runtime: Python 3.12, ONNX Runtime 1.30.0, Tokenizers 0.23.2, NumPy 2.5.3, Pillow 12.3.0 and Tesseract 5.5.3. Model sources: [Google SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224), [pinned ONNX conversion](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/tree/ba1f3b0843f24bc5417d38e19c37b287d719b2f4). Model data is Apache 2.0 licensed; example attribution is in `CREDITS.md`.
