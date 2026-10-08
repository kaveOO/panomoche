# Using panomoche

Reference for the web app, the commands and the day-to-day details. For the overview, see
the [README](../README.md); for the design, [ARCHITECTURE.md](ARCHITECTURE.md); for the
reasoning behind every rule, [METHODS.md](METHODS.md).

* [Rules and principles](#rules-and-principles)
* [Results](#results)
* [Tech stack](#tech-stack)
* [Using the web app](#using-the-web-app)
* [Commands](#commands)
* [Things to know](#things-to-know)
* [Development](#development)
* [Project layout](#project-layout)
* [Licences](#licences)

## Rules and principles

All measurements are taken first. Then a single function, `panomoche/decision.py`, applies
the rules below in order. The first rule that matches excludes the picture. If none
matches, the picture is kept.

| Reason | Excluded when | Measured by |
|---|---|---|
| `low_resolution` | fewer than 15 px per degree of view | Panoramax metadata |
| `advertisement` | layout, text area and commercial wording all point to an advertising graphic | SigLIP 2 + Tesseract OCR |
| `privacy_blur` | blurred faces and plates cover more than 5% of the picture | SGBlur's record in the original file |
| `wrong_orientation` | upside down or sideways with ≥ 99% confidence (360°: 2 of 4 rendered views rotated) | DINOv3 + logistic regression |
| `not_sharp` | ≥ 33% of the textured area blurred with CPBD < 0.15, or ≥ 20% with CPBD < 0.12 | OpenCV (detail ratio, edge widths) |
| `wavy` | rolling-shutter wobble, confirmed by ≥ 30% of its sequence neighbours (flat pictures only) | Canny edge bending |
| `sequence_outlier` | TOPIQ at least 0.12 below the median of the 10 pictures before and after | TOPIQ-NR |
| `low_quality` | TOPIQ under 0.32 (360° or in a sequence), or under 0.40 (a flat picture judged alone) | TOPIQ-NR |

**Principles**

* **Exclude only what is clearly bad.** Every threshold was set by looking at all the
  pictures it catches on real Panoramax data. When in doubt, the picture is kept.
* **Judge a picture against its own sequence.** Quality scores mostly rank cameras: a
  dashcam behind a windshield is softer than a phone on every picture. Comparing a picture
  with its neighbours finds the ones that are worse than their own camera's normal. A
  picture judged alone gets the benefit of the doubt.
* **No manual override.** The rules alone decide: there is no button to keep or exclude a
  picture by hand.
* **One system everywhere.** `predict`, the upload page and the review page all call the
  same `decide` function. Uploading a picture that is on the review page gives the same
  verdict.

## Results

| Dataset | Pictures | 360° | Cameras | Kept | Excluded |
|---|---:|---:|---:|---:|---:|
| `data/all.jsonl` (tuning set) | 1,300 | 491 | 102 | 1,158 | 142 |
| `data/new.jsonl` (unseen) | 493 | 128 | 97 | 477 | 16 |
| `data/new2.jsonl` (unseen) | 590 | 135 | 100 | 573 | 17 |

* **Tuning set:** 4 sequences of 150 pictures, plus 700 random and Paris pictures. The
  142 exclusions: 47 not sharp, 46 wavy, 17 low quality, 14 wrong orientation, 10 sequence
  outliers, 7 privacy blur, 1 low resolution.
* **Unseen sets:** random pictures, one per sequence, fetched after the rules were tuned.
  They share no picture or sequence with the tuning set, so they show how the rules behave
  on new data. Exclusions: 14 not sharp and 2 low quality, then 13 not sharp and 4 low
  quality.

### Blur benchmark

The benchmark takes 200 unseen pictures, blurs them 6 different ways, and sets every method
to the same 3.5% false-alarm rate. Full report:
[reports/blur_benchmark.md](../reports/blur_benchmark.md).

| Blur added | panomoche rule | Laplacian variance | TOPIQ |
|---|---:|---:|---:|
| focus σ 1.5 px / 3 px | 100% / 100% | 100% / 100% | 36% / 95% |
| motion 5 / 11 / 21 px | 56% / 74% / 73% | 36% / 78% / 91% | 28% / 85% / 100% |

The rule takes 23 ms per picture. Motion blur is its weak spot (see [Known limits](#known-limits)).

## Tech stack

| Area | Technology | Used for |
|---|---|---|
| Language | Python 3.14, managed with [uv](https://docs.astral.sh/uv/) | everything |
| Image quality | [TOPIQ-NR](https://github.com/chaofengc/IQA-PyTorch) (SPAQ weights), `pyiqa` 0.1, PyTorch 2 on CPU | quality score, sequence comparison |
| Vision backbone | [DINOv3](https://github.com/facebookresearch/dinov3) ViT-S/16 via `timm` 1.0 | orientation classifier, rain experiment |
| Vision-language | [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) base, quantized ONNX, `onnxruntime` 1.30, `tokenizers` | advertisement layout |
| OCR | Tesseract 5 (English) | advertisement wording |
| Classic vision | OpenCV 5, NumPy 2, Pillow 12 | sharpness, waviness, 360° projections |
| Machine learning | scikit-learn 1.9, joblib | orientation classifier, evaluations |
| Web | FastAPI, Uvicorn, plain HTML/CSS/JavaScript (no framework, no CDN) | upload page, review page, HTTP API |
| Data source | Panoramax STAC API (`requests`), HTTP Range reads | sampling, downloads, SGBlur records |
| Quality | pytest (169 tests), httpx, ruff, vulture, Playwright | tests, lint and format, dead code, UI checks |

Everything runs locally on CPU. No picture leaves the machine; the only network traffic is
the downloads from Panoramax.

## Using the web app

`panomoche serve --predictions <file.jsonl>` starts a local server on `127.0.0.1:8000`.

### Upload page (`/`)

Drop one picture or a whole sequence. Each picture shows:

* its verdict and reason;
* its measurements, with a per-view breakdown for 360° pictures;
* the privacy blur boxes;
* the tags that `tag --apply` would write.

With 6 pictures or more, they are also compared as one sequence, in file-name order.

### Review page (`/review`)

Shows every picture of the predictions file, with:

* a Kept / Excluded filter, and one filter per reason;
* a per-sequence table and sequence order;
* a full-size viewer (use ← and → to browse).

### HTTP API

| Endpoint | Does |
|---|---|
| `POST /predict` | judge one picture (multipart `picture`, optional `sgblur` header from SGBlur) |
| `POST /api/sequence` | judge measured pictures together as one sequence |
| `GET /api/info` | model, thresholds and enabled checks |

`POST /predict` is what a Panoramax backend would call on upload, the same way it already
calls SGBlur for privacy blurring.

## Commands

| Command | What it does |
|---|---|
| `panomoche fetch` | download a sample (`--random N` or `--bbox`), skipping known pictures and removing duplicates |
| `panomoche predict` | measure and sort local pictures, a bbox or a sequence (`--collection`); `--html` writes a static review page |
| `panomoche serve` | run the upload page, review page and HTTP API |
| `panomoche tag` | write results to Panoramax as semantics tags; **dry run unless `--apply`** |
| `panomoche waviness` / `waviness-ui` | the merged Waviness Check: batch reports with overlays, or its own UI |
| `panomoche ads` / `ads-ui` / `ads-setup` | the merged Advertisement Check: batch reports, its own UI, or the model download |

Every command has `--help`. Useful options: `--no-sequence`, `--no-orientation`,
`--no-waviness`, `--no-ads`, `--max-blurred-area`, `--max-privacy-blur`.

## Things to know

### Running it

* **Memory:** built and tested on a CPU-only machine with 8 GB of RAM (WSL2). The server
  holds TOPIQ, DINOv3 and SigLIP 2, about 2 to 3 GB. LIQE, one of the compared models, needs
  more than 6 GB on first load, so it is not used by default.
* **Speed:** the full `predict` pipeline takes about 1 to 1.5 s per picture on CPU. OCR alone
  adds about 0.5 s.
* **Tesseract version:** the advertisement check was verified with Tesseract 5.5. Ubuntu
  ships 5.3, which reads stray `"` characters. A parsing fix in `advertising/classifier.py`
  makes both work.
* **WSL2:** the server listens on `127.0.0.1`. `http://localhost:8000` normally works from
  Windows. If not, use `--host 0.0.0.0`, which also exposes the server to your network.
* **Negative coordinates:** `--bbox` needs an `=` sign, as in `--bbox=-1.2,47.1,-1.0,47.3`.

### Panoramax specifics

* Pictures are already anonymised by SGBlur when downloaded. SGBlur stores its detections
  as a comment at the end of the *original* (hd) file only, and panomoche reads them with
  HTTP Range requests. The IGN instance does not support Range requests, so use
  `--privacy-full-download` there.
* The "sd" asset is always 2048 px wide, so small originals arrive enlarged. Sharpness is
  measured at the original size (read from the sensor metadata), so that these pictures are
  not called blurred.
* **Nothing is written to Panoramax** unless `panomoche tag --apply` is run with a token.

### Known limits

* **Motion blur:** the rule catches 73% of strong motion blur, against 100% of focus blur.
  Smear that keeps detail across the motion direction (handlebar and helmet mounts, the LG
  360° camera) often measures as sharp.
* **Rain on the lens:** drops are local soft patches inside a sharp scene, so the blur rule
  lets these pictures through. A DINOv3 classifier recognises them: it flags 30 of 30 rainy
  frames of an unseen sequence and 0 of 30 clean pictures (`scripts/rain_model_eval.py`).
  However, it has only seen two rainy sequences, from one day and one camera, so it is not
  part of the decision yet. See "Rain on the lens" in METHODS.
* **Uniform sequences:** when a whole sequence has the same fault (shot through a side
  window, an obstruction in the corner), no picture stands out from its neighbours.
* **Waviness** is only judged on flat pictures, and only when the sequence confirms it. A
  wide-angle lens, a curved building or tree trunks can bend the lines of a single picture
  too.
* **Orientation** catches rotations of 90° or more, not a slightly tilted horizon.
* **Thresholds** were tuned on French Panoramax pictures and checked by eye. They are not a
  calibrated accuracy on a labelled benchmark.

### Merged tools

`waviness/` and `advertising/` come from two standalone tools delivered as zip archives.
Their code stays close to upstream, so that a new release can be merged again, and their own
tests run in the suite (`tests/test_waviness_tool.py`, `tests/test_advertising_tool.py`).

Changes made during the merge:

* package imports;
* model location (`models/siglip2`);
* no global Pillow size limit;
* the Tesseract parsing fix.

Their original READMEs and validation records are in
[merged-tools/](merged-tools/README.md).

## Development

```bash
uv run pytest                                          # 169 tests, about 1 minute
uv run ruff check && uv run ruff format --check        # lint and format
uv run vulture panomoche scripts --min-confidence 60   # dead code (FastAPI routes are expected hits)
uv run python scripts/blur_benchmark.py --n 200        # blur benchmark, about 20 minutes
```

The advertisement tests need the SigLIP 2 model and Tesseract.

## Project layout

```text
panomoche/
  cli.py              the `panomoche` command
  decision.py         the single decision: excluded (with a reason) or kept
  sharpness.py        blurred area (detail ratio) and CPBD (edge widths)
  orientation.py      rotation classifier on DINOv3 features, 360° rendered views
  privacy.py          privacy blur share from SGBlur's record (also over HTTP Range)
  metadata.py         resolution and GPS checks from Panoramax metadata
  iqa.py              TOPIQ, MUSIQ and LIQE via pyiqa, per-view scores for 360°
  scoring.py          TOPIQ scorer factory
  sequence.py         per-sequence summaries
  dedupe.py           known-picture skipping, duplicate and near-duplicate removal
  panoramax.py        Panoramax API client, sampling, semantics payloads
  service.py          FastAPI app (upload, review, API)
  report.py           review page rendering (served or static)
  imaging.py          loading, 360° detection, views
  backbone.py         DINOv3 feature extractor (orientation, rain experiment)
  waviness/           merged Waviness Check v0.2 (detector, UI)
  advertising/        merged Advertisement Check v0.4 (SigLIP 2 + OCR classifier, UI, policy)
  web/                upload.html, review.html
scripts/              evaluations and benchmarks (each one documents itself)
tests/                pytest suite
examples/             reference pictures of the merged tools (credits in examples/advertising)
docs/
  ARCHITECTURE.md     synthesis, C4 diagrams, API, data
  METHODS.md          detailed methods, measurements and decisions
  merged-tools/       original documentation of the two merged tools
reports/              benchmark reports
data/, models/        downloaded pictures, results and model files (not versioned)
```

## Licences

* TOPIQ, MUSIQ and LIQE weights are downloaded by pyiqa. Check their licences before
  redistributing them.
* DINOv3 weights are under the
  [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
* SigLIP 2 is under Apache 2.0. Sources and licences of the example pictures are in
  `examples/advertising/CREDITS.md`.
* Pictures downloaded from Panoramax keep their licence, mostly CC-BY-SA-4.0 or etalab-2.0,
  recorded in each `meta.jsonl`. Credit the authors if you redistribute them.
