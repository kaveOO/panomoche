# panomoche

**Automatic sorting of street-level pictures for [Panoramax](https://panoramax.fr).
Keep the clean pictures, exclude the unusable ones, and say why.**

Contributors upload pictures to Panoramax from phones, dashcams, helmets and 360° cameras.
Some are blurred, rotated, smeared by a shaking mount, hidden by privacy blurring, or not
street pictures at all. panomoche measures every picture, compares it with the rest of its
sequence, and sorts it as **kept** or **excluded**, with one reason.

## How a picture is judged

A single function (`panomoche/decision.py`) applies these rules in order. The first match
excludes the picture; if none matches, it is kept.

| Reason | Excluded when |
|---|---|
| `labelled_blurred` | a reviewer marked it Blurred |
| `low_resolution` | fewer than 15 px per degree of view |
| `advertisement` | it is an advertising graphic (SigLIP 2 + OCR) |
| `privacy_blur` | blurred faces and plates cover more than 5% of it |
| `wrong_orientation` | upside down or sideways (DINOv3) |
| `not_sharp` | a large part of the detailed area is blurred (OpenCV, CPBD) |
| `wavy` | rolling-shutter wobble, confirmed by its sequence |
| `sequence_outlier` | clearly worse than the 10 pictures before and after (TOPIQ) |
| `low_quality` | very low TOPIQ quality score |

When in doubt, a picture is kept. On two sets of unseen pictures, about **97% were kept**.

## Tech stack

Python 3.14 and uv · TOPIQ (pyiqa), DINOv3 (timm), PyTorch on CPU · SigLIP 2 (ONNX Runtime),
Tesseract OCR · OpenCV, NumPy, scikit-learn · FastAPI · Panoramax API.

## Quickstart

```bash
uv sync --all-extras                    # Python 3.14 and every dependency in .venv
sudo apt install tesseract-ocr          # OCR for the advertisement check
uv run panomoche ads-setup              # advertisement model, about 393 MiB (once)
```

```bash
uv run panomoche fetch --random 500 --new-sequences --out data/new   # get pictures
uv run panomoche predict data/new --out data/new.jsonl               # sort them
uv run panomoche serve --predictions data/new.jsonl                  # http://localhost:8000
```

The server opens an upload page to test your own pictures, and a review page to browse the
results and mark blurred pictures. `POST /predict` is the endpoint a Panoramax server would
call on upload. **Nothing is written to Panoramax** unless `panomoche tag --apply` is run.

The orientation check needs a model trained once from upright pictures (about 10 minutes, no
labels): `uv run panomoche fetch --out data/raw`, then `uv run python scripts/orientation_eval.py`.

## Documentation

* [docs/USAGE.md](docs/USAGE.md): web app, commands, things to know, known limits.
* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): C4 diagrams, API and data.
* [docs/METHODS.md](docs/METHODS.md): exact thresholds, measurements and rejected ideas.
* [reports/blur_benchmark.md](reports/blur_benchmark.md): blur detection benchmark.

## Licences

Model weights keep their own licences (pyiqa models, the
[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/), Apache
2.0 for SigLIP 2). Pictures from Panoramax keep theirs, mostly CC-BY-SA-4.0 or etalab-2.0:
credit the authors if you redistribute them.
