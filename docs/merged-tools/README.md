# Merged tools: original documentation

panomoche includes two standalone tools, delivered as `r` zip archives and merged into the
package. Their code now lives in `panomoche/waviness/` and `panomoche/advertising/`, their
reference pictures in `examples/`, their tests in `tests/test_waviness_tool.py` and
`tests/test_advertising_tool.py`, and the advertisement model in `models/siglip2/`.

This folder keeps what the archives contained besides code, for reference:

| File | What it is |
|---|---|
| `waviness-check.md` | the Waviness Check v0.2 README: method, scoring formula, validation and limits |
| `advertisement-check.md` | the Advertisement Check v0.4 README: decision method, HTTP API, limits |
| `*-validation.json` | each tool's recorded test and example results at delivery |
| `make_advertising_fixtures.py` | how the synthetic advertisement test pictures were generated (paths as in the original archive; kept for provenance, not run by panomoche) |

Paths and commands in these READMEs refer to the standalone layout (`./start.sh`,
`python detector.py`, `python -m unittest test_filter`). In panomoche use instead:

| Standalone | panomoche |
|---|---|
| `python detector.py …` | `panomoche waviness …` |
| `./start.sh` / `python server.py` (waviness) | `panomoche waviness-ui` |
| `python classifier.py …` | `panomoche ads …` |
| `./start.sh` / `python server.py` (advertisement) | `panomoche ads-ui` |
| `python setup_models.py` | `panomoche ads-setup` |
| `python -m unittest test_detector` / `test_filter` | `pytest tests/test_waviness_tool.py tests/test_advertising_tool.py` |

Changes made when merging (all visible as small diffs against the originals): package
imports, paths to `web/` and `examples/`, the model location, `main(argv)` entry points for
the `panomoche` command, no global Pillow image-size limit (it blocked reading large 360°
originals), and a Tesseract TSV parsing fix (a stray `"` read by Tesseract 5.3 hid the words
after it).

Reference pictures in `examples/`: five oversized photos were resized to keep the repository
light (40 MB → 14 MB): `waviness/1.png`, `2.png`, `cows.png` and their copies
`advertising/street-wavy.png`, `animals.png` to 1,280 px (the size the waviness detector
analyses anyway, with the same OpenCV resampling), and `advertising/park-statue.jpg`,
`ice-landscape.jpg` to 2,048 px. Both tools' tests pass unchanged, and the waviness scores
are the published ones (84.8, 51.5, 0.0, 6.9).
