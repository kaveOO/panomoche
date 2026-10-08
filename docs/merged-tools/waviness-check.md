# Waviness Check

A local image tool for detecting repeated geometric bending in long, mostly vertical edges. Includes a browser UI and a Python batch CLI. Images stay on the computer; there are no cloud calls or CDN dependencies.

## Use the running tool

Open **http://127.0.0.1:8765**. Choose images or click **Try the four reference images**. Select a result, inspect **Evidence**, **Original**, or **Edge map**, and export the PNG overlay or JSON report. Multiple uploads are processed sequentially. Images must be at least 96 pixels on each side; the UI accepts files up to 20 MB and the decoder limits images to 40 megapixels.

Version 0.2 with default settings produces these results:

| Example | Waviness score | Label | Evidence confidence |
|---|---:|---|---|
| Wavy street 1 | 84.8 / 100 | Waviness detected | High |
| Wavy street 2 | 51.5 / 100 | Waviness detected | Medium |
| Cows | 0.0 / 100 | Insufficient evidence | Low |
| Straight street | 6.9 / 100 | No strong waviness evidence | Medium |

The score is a heuristic index, **not a probability**. Evidence confidence describes the number and independence of usable supporting edges; it is not a measured accuracy rate.

The `sample-results/` folder contains saved reports, overlays, and edge maps for all four examples. `preview.jpg` shows the browser interface. Browser exports are served as normal local file downloads; temporary export bytes are kept in memory for up to 15 minutes, with a limit of 16 recent exports.

## Restart on this computer

From this tool directory:

```sh
./start.sh --port 8765
```

The launcher uses the prepared environment in this workspace. You can also run `/home/NotArt/Documents/Codex/2026-10-06/s/work/.venv/bin/python server.py --port 8765` directly.

The process binds to `127.0.0.1`. Press Ctrl+C in its terminal to stop it. Use another port if 8765 is occupied.

## Install on another computer

Requires Python 3.10 or newer. From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python server.py
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`. Open the local URL printed in the terminal.

## Batch command line

```sh
python detector.py image1.jpg image2.png --output-dir results
python detector.py pictures/*.jpg --output-dir results --threshold 55 --min-bending 0.25
```

Each input produces JSON, an evidence overlay, and a Canny edge map. A numeric prefix prevents images with matching stems from overwriting each other within a batch. Invalid inputs are reported, valid inputs continue, and the command exits with status 1 if any input failed. Existing output names from a previous run are replaced; choose a new output directory to preserve previous runs.

To integrate directly:

```python
from pathlib import Path
from detector import Settings, analyze, load_image

image = load_image(Path("image.jpg").read_bytes())
report, resized, overlay, edge_map = analyze(image, Settings())
print(report["label"], report["score"], report["confidence"])
```

## Method

1. Apply EXIF orientation and resize to a maximum side of 1280 pixels, without upscaling.
2. Use Sobel gradients to select contrast-adaptive Canny extraction thresholds.
3. Split contours into long, mostly vertical chains; remove duplicate traces.
4. Smooth each chain's sideways position along image rows. Measure displacement after subtracting a fitted straight line. Subtract a quadratic baseline for a second measure that suppresses simple bowing.
5. Retain chains whose nonlinear bending exceeds the configured fraction of image width and the tracing noise floor.
6. Require confirming traces to span at least 15% of image height, with a common section spanning at least 10% of height and 60% of the shorter trace. Compare separation at the same image rows. Compare the actual line-detrended profiles, rather than quadratic residuals that can artificially correlate on short arcs. Correlation may have either sign because rotation can produce opposite motion in different regions.
7. Compute shared edge coverage: the length of confirmed traces divided by total usable trace length. Combine this with median nonlinear amplitude, profile agreement, and spatial support into a 0–100 index. At least three separated support groups are required for the `wavy` label. Straight reference edges now reduce the image-level score.

Coral = bending corroborated by another separated trace. Amber = bending without corroboration. Blue = a usable reference edge. Numbered markers match the JSON edge IDs and profile chart. Overlays are exported at analysis resolution, recorded in the report.

For corroborated traces, the score is:

```text
100 × min(1, median_bend / (2.2 × minimum_bend))
    × min(1, independent_groups / 3)
    × median_absolute_correlation
    × min(1, shared_edge_length_fraction / 0.30)
```

Without corroboration, the score is zero. With fewer than three spatial support groups, it is capped at 30. With fewer than six long traces and fewer than three support groups, the result is `insufficient_evidence`; this should not be treated as an automatic rejection. The default nonlinear bending floor is 0.20% of image width, the correlation threshold is 0.78, and the decision score is 45. The UI records each result's settings and requires reanalysis to apply changes.

Labels are `wavy`, `review`, `no_strong_evidence`, and `insufficient_evidence`. No-strong-evidence does not prove an image is geometrically correct.

## Validation and limits

Run the included checks:

```sh
python -m unittest -v test_detector
```

The 21 tests pass in the prepared environment. Tests cover synthetic straight facades, perspective changes, quadratic bowing, blur alone, shared sinusoidal warps, warps with blur, sparse scenes, random texture, unrelated curves, a single curved structure, the supplied wavy examples, both real false-positive examples, the straight street at two resolutions, a few curves among many straight references, reported edge coverage, EXIF orientation, invalid inputs, and the local HTTP API. These supplied photos are development regression cases, not an independent accuracy benchmark.

This is an interpretable **prototype**, not a detector calibrated on a real camera dataset. It assumes long edges correspond to structures that should be straight. Vegetation, repeated curved architecture, genuinely curved rails, and complex lens distortion can create false positives. Horizontal-only or rotated scenes, short contours, low contrast, and missing edges can create false negatives. Removing a quadratic baseline can miss a distortion visible as only a single broad bow. It does not identify the physical cause or repair the image. Blur is not scored, although it can weaken edge evidence.

Before using the tool to automatically reject frames, label representative normal and wavy images from the same camera, split validation by capture sequence, then tune the minimum bending and decision threshold against the desired false-positive and false-negative rates. For vegetation-dominated scenes, neighboring video frames or camera motion data would provide additional evidence.

Method references: [OpenCV Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html), [OpenCV Sobel](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html), [Removing Rolling Shutter Wobble](https://www.microsoft.com/en-us/research/publication/removing-rolling-shutter-wobble/). The scoring rule here is custom; those references do not establish its accuracy.

Dependencies used for verification: Python 3.12, OpenCV 5.0.0, NumPy 2.5.3, Pillow 12.3.0. `requirements.txt` supports a broader version range; other combinations have not been verified.
