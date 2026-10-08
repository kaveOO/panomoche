# panomoche: methods and research notes

This is the detailed record behind the decisions summarised in the [README](../README.md):
every measurement, how each threshold was chosen on real Panoramax pictures, and what was
tried and dropped. Numbers refer to the datasets at the time each section was written.

## How pictures are sorted

Every picture is measured first (Panoramax metadata, privacy blur, orientation,
sharpness, advertisement, waviness, TOPIQ). Then one decision, in `panomoche/decision.py`, sorts it as
**excluded** or **kept**. There is no manual override, and the aim is to
exclude only pictures that are really not clean. A picture is excluded, with
one reason, when any of these applies:

1. **Whatever the context**
   * `low_resolution`: below 15 px per degree;
   * `advertisement`: an obvious advertising graphic (see "Waviness and
     advertisement checks");
   * `privacy_blur`: Panoramax's blurred faces/plates cover more than 5% of the picture;
   * `wrong_orientation`: upside down or sideways, with at least 99% confidence
     (360° pictures: a camera mounted pitched or rolled 90°);
   * `not_sharp`: two sharpness measures agree **at viewing size** (1024 px
     wide, or the original size if smaller): 33% or more of the textured area
     blurred with a CPBD (share of edges that look sharp) under 0.15, or 20%
     or more with a CPBD under 0.12. Pixel-level softness at 100% zoom (phone
     processing, compression) doesn't count. Panoramax's 2048 px "sd" files of
     small originals are enlarged, and measuring them as they come made a fifth
     of a random sample look blurry.
2. **Rolling-shutter wobble**
   * `wavy`: a flat picture whose straight edges bend together, when at least
     30% of its sequence neighbours are wavy too (see "Waviness and
     advertisement checks").
3. **Image quality (TOPIQ)**
   * `sequence_outlier`: TOPIQ 0.12 or more below the median of its
     neighbours (at least 2 pictures of the same sequence within 10 positions);
   * `low_quality`: TOPIQ below 0.32 (360° pictures, or any picture with
     neighbours) or below 0.40 (a flat picture alone, given the benefit of the doubt).

Every exclusion on 1,300 pictures was checked by eye, and the limits
were set to keep pictures that look acceptable:

* At viewing size, a dashcam's windshield haze stays around 14% blurred while
  real motion blur reaches 35%, so one blur limit works for every camera.
* Just above the TOPIQ floors are dim dusk 360° pictures and plain country
  roads that look fine.
* Soft but clean textures (foliage, dusk, small phone-app pictures of a
  windmill, a chapel, sunsets) can reach 33-38% blurred area. CPBD looks only
  at edges, finds them sharp, and keeps these pictures.

Reviewer's examples, judged with their sequence: 6 of 7 as wanted. To exclude:
forest in sun glare, washed-out van, blurry dusk picture, motion-blurred
handlebar picture. To keep: a 360° picture, a dashcam sequence through a hazy
windshield. The exception is a shot through a side window. Its whole sequence
was shot that way, and nothing measured separates it from the dashcam (same
softness, same blur, and the phone's heading points along the road). It needs
a model trained on reviewed examples of what pictures show.

### Why compare with the sequence

TOPIQ and sharpness scores mostly rank **cameras**. On 310 held-out Panoramax
pictures, 15 of the 24 lowest TOPIQ scores came from one camera model (GoPro
Fusion). A dashcam behind a windshield is softer than a phone on every
picture. Comparing each picture with the 10 before and after it (ordered by
rank in the collection) removes that bias and finds what matters: a picture
clearly worse than the ones taken just before and after it.

`--collection` is the natural input, because a bbox sample has one picture per
sequence and is judged with the absolute limits. On long sequences,
`--start-after-rank N` starts further in. It only works against the instance
hosting the sequence (`--api https://panoramax.openstreetmap.fr/api`): the
federated meta-catalog ignores it, and the client raises an error instead of
silently scoring the wrong pictures.

### New dataset, without repeats

`fetch` never brings back a picture already judged: it skips every id in
`--skip-known` (by default `data/all.jsonl`). With
`--new-sequences`, it also skips sequences already sampled. It removes
duplicates: the same file, or near-identical pictures (64-bit difference hash,
at most 6 bits apart), such as a camera shooting while stopped at a light or a
picture uploaded twice. Each removed duplicate is listed in `duplicates.jsonl`
with the picture it repeats.

```bash
panomoche fetch --random 500 --seed 11 --new-sequences --out data/new
panomoche predict data/new --out data/new.jsonl
```

That run gave 493 new pictures from 493 different sequences, taken with 97
camera models. It removed 6 duplicates and shares no picture with the first
1,300. The verdicts: 477 kept, 14 not sharp, 2 low quality.

### Results

On 1,300 pictures (4 sequences of 150, plus 700 random and Paris pictures
judged alone): 1,158 kept, 142 excluded (47 not sharp, 46 wavy, 17 low
quality, 14 wrong orientation, 10 sequence outliers, 7 privacy blur, 1 low
resolution, 0 advertisements).

The floor for a flat picture alone (0.45 then, 0.40 now) is only a fallback for samples. The
13 random pictures it caught were judged again with their 20 sequence
neighbours, which is how Panoramax would judge them: 5 came out kept. These
were dull cameras whose whole sequence scores around 0.42, such as a hazy
country road. The other 8 stayed excluded, as outliers or as dim 360° pictures.
Judging by sequence has a known blind spot: when a whole sequence has the same
fault (a side-window shot, an obstruction in the corner), no picture stands out.

### Pictures are already anonymised

Every picture fetched from Panoramax has been through SGBlur, which blurs
faces and plates inside JPEG blocks: hard-edged rectangles, the rest of the
picture untouched. A blur-sensitive model could take those patches for a
defect, so `scripts/privacy_blur_sensitivity.py` adds more SGBlur-like
patches to 60 held-out pictures and measures the TOPIQ drop:

| added patches                     | median drop | 90th pct | max   | drops ≥ 0.10 |
|-----------------------------------|------------:|---------:|------:|-------------:|
| 6 small (distant faces, plates)   | 0.000       | 0.002    | 0.005 | 0 / 60       |
| 1 large (close pedestrian)        | 0.000       | 0.011    | 0.064 | 0 / 60       |
| 2 large                           | 0.007       | 0.030    | 0.058 | 0 / 60       |

TOPIQ barely reacts. The one flagged picture that has a large real patch
(#308 of the Sony sequence) confirms it: pasting a patch of the same size and
position onto its six neighbours changes their scores by −0.016 to +0.007,
and what lowers #308 is the rest of the upper picture (a motion-blurred
pedestrian, a softer facade).

So for the Panoramax integration, the TOPIQ check can run before or after
SGBlur with practically the same result. This was measured for TOPIQ only:
the DINOv3 classifier's handcrafted sharpness signals (worst block, share of
flat blocks) are likely more sensitive and would need the same check. Its
training pictures were also anonymised, so patches appear in its "good"
examples, which should teach it to ignore them.

```bash
python scripts/privacy_blur_sensitivity.py data/eval --n 60
```

### Excluding heavily anonymised pictures

Blurring doesn't distort the score, but it hides part of the scene. So
`fetch`, `predict` and `serve` exclude pictures where blurred faces and plates
cover more than `--max-privacy-blur` (default 5%) of the picture. For 360°
pictures this is measured over the whole picture too: measuring only the
horizon band counted the patches twice and excluded clean 360° pictures. An excluded
picture is not scored, not tagged, and not used as a neighbour in the sequence
comparison. The review page has an "excluded" filter, and the upload page
draws the blurred boxes.

**Where the numbers come from.** SGBlur appends what it detected to the
original JPEG as a comment at the very end of the file:

```
[{'class': 'face', 'confidence': 0.066, 'xywh': [448, 0, 416, 384]}]
```

Panoramax doesn't expose this in its API, and the re-encoded `sd` and `thumb`
versions lose it, but the `hd` asset is the original file and keeps it.
Because the record is at the end of the file, two HTTP range requests (the
last 64 KB for the record, the first 64 KB for the picture size) are enough:
600 pictures take about 25 s. Only faces and plates of at least 12 px count,
since SGBlur doesn't blur road signs or smaller boxes. A picture without a
record had nothing detected, because SGBlur only writes one when it finds
something. When the share can't be read, the picture is kept.

* **Range support varies by instance.** OSM-FR serves ranges. IGN didn't when
  checked (October 2026) and would send whole originals of several MB, so its
  pictures stay "unknown" unless you pass `--privacy-full-download`.
* **On upload,** a Panoramax backend already has the same detections in
  SGBlur's `x-sgblur` response header, and can pass them to `/predict` as the
  `sgblur` form field.
* **Local files** only carry the record if they are Panoramax originals (the
  `hd` download). The upload page reads it from such files.

On the 600 pictures of the four test sequences, half have some blurring,
usually tiny (90th percentile 1.4% of the scored area). 15 exceed 5% and 5
exceed 10%, all GoPro Fusion pictures of a busy street. In those, the large
boxes are not faces at all: SGBlur blurs parts of facades, windows and shop
signs on low-confidence guesses. Across the 1,096 blurred boxes, **85% of the
blurred area comes from detections below 25% confidence**, and the largest
boxes have a median confidence of 0.08. This is presumably a deliberate
privacy-first setting, but it is worth knowing: heavy anonymisation mostly
means "SGBlur was unsure", not "many people".

Detecting the patches from pixels alone was tried first and abandoned. Most
cameras deliver pictures as smooth as a blurred patch (noise reduction,
compression, stitching), so a pixel detector found only 14 of 40 large
synthetic patches, and the full-resolution file only helped for one grainy
camera.

## Panoramax integration

There are three ways to plug this into Panoramax. They don't exclude each other.

1. **External bot, no backend change (available now).** `predict` then `tag`
   writes the issues as semantics tags on each picture. The GeoVisio API (2.16)
   lets any authenticated account add tags with
   `PATCH /api/collections/{cid}/items/{id}`. Writes go to the *hosting*
   instance (IGN, OSM-FR...), resolved from the meta-catalog `via` link.
   `tag` is a dry run unless you pass `--apply`:

   ```bash
   panomoche tag seq.jsonl                                # prints the PATCH requests
   PANORAMAX_TOKEN=... panomoche tag seq.jsonl --apply     # writes them
   ```

   Tags follow the qualifier convention already used by the traffic-sign bot
   (`detection_model[osm|traffic_sign=yes]=SGBlur-yolo11s/0.1.0`):

   ```
   quality_issue=sequence_outlier
   detection_model[quality_issue=sequence_outlier]=panomoche-topiq_nr-spaq/0.1.0
   ```

   Classifier issues also get
   `detection_confidence[quality_issue=blur]=0.973`, and `tag --min-confidence`
   (0.9) filters on it. TOPIQ and sequence issues come from thresholds, not
   probabilities, so they get no confidence tag and are written as decided by
   `predict`.

   Panoramax has no standard quality tag yet. `quality_issue` is a proposal
   (no prefix = Panoramax-own tag) and should be agreed with the community
   before anything is written at scale. The key can be changed with `--tag-key`.

2. **On-upload scoring.** GeoVisio already sends pictures to an external SGBlur
   HTTP service for face and plate blurring. A quality service fits the same
   pattern:

   ```bash
   panomoche serve --port 8000
   curl -F picture=@photo.jpg http://localhost:8000/predict
   ```

   The response contains the score, the issues, per-view scores and a
   ready-to-store `semantics` list. A single uploaded picture has no sequence
   context, so on upload only the absolute check applies. The sequence
   comparison (`POST /api/sequence`, or `predict --collection`) fits better as
   a job run once a sequence is complete.

3. **Moderation reports.** The API already has a `picture_low_quality` report
   type. `tag --apply --report` also opens a report whose comment gives the
   score and the sequence median. Use it sparingly: automated reports land in
   moderators' queues.

### Excluding pictures that aren't sharp

`fetch`, `predict` and `serve` also exclude pictures when **two sharpness
measures agree** (`--max-blurred-area`, default 0.33, 1 disables it;
`--max-cpbd`, default 0.15). See `panomoche/sharpness.py`.

The first is the *blurred area*. The picture is cut into 64 px blocks at
1024 px width. Flat blocks (sky, plain walls) are left out, and a
textured block counts as blurred when its fine detail is low compared with its
medium-scale detail (ratio below 0.25). Privacy-blur boxes are ignored.

`scripts/sharpness_eval.py` measures how well the detail ratio separates sharp
from blurred pictures (1.0 = perfect, 0.5 = chance):

| blur applied to 60 held-out pictures | detail ratio | TOPIQ |
|--------------------------------------|-------------:|------:|
| mild Gaussian (σ = 1 px)             | 1.00         | 0.60  |
| strong Gaussian (σ = 2.5 px)         | 1.00         | 0.78  |
| motion (9 px)                        | 0.91         | 0.75  |

A whole-picture average was tried first and missed partly blurred pictures.
On Sony #303 (handlebar phone), a sharp distant street in the middle hid
motion blur and wobble over the whole foreground: average 0.31, above the
cutoff, but 53% of the textured area blurred.

On the 600 sequence pictures, the rule excludes 30: 22 handlebar-phone
shots with motion blur and wobble, and 8 dashcam shots through a hazy
windshield. The ones checked at full resolution are all visibly soft. 360°
cameras have a median blurred area of 4%. Comparing sharpness with
neighbours was tried and isn't used: plain walls and sky have little fine
detail, so half of the pictures it singled out were not blurry.

The second is **CPBD** (Cumulative Probability of Blur Detection, Narvekar &
Karam 2011). It measures the width of every vertical edge (the span of the
monotonic intensity run across it) and gives the share of edges whose blur a
viewer wouldn't notice: up to 5 px wide at low contrast, 3 px at high
contrast. It looks only at edges, so soft textures don't count.
`scripts/cpbd_eval.py` compares it at 1024 px, the same blur copies as above:

| blur applied to 60 held-out pictures | blurred area | CPBD | variance of Laplacian |
|--------------------------------------|-------------:|-----:|----------------------:|
| mild Gaussian (σ = 1 px)             | 1.00         | 1.00 | 0.99                  |
| strong Gaussian (σ = 2 px)           | 1.00         | 1.00 | 1.00                  |
| motion (7 px)                        | 0.93         | 1.00 | 0.94                  |

On real pictures, CPBD alone calls clean foliage, church and Paris street
pictures blurry, so it isn't used alone. As a confirmation it changes, on
1,300 pictures checked by eye:

* **Kept** (excluded before): a windmill, a cemetery chapel, a dark mountain view, two sunsets, a
  360° street, a Paris street, a soft dashcam picture.
* **Excluded** (kept before): 17 handlebar pictures with rolling-shutter smear
  (wobbly shopfronts), the reviewer's washed-out van, a picture through a dirty
  side window, a dark dusk picture.
* **Mistakes**: excluded a picture of ducks and a Paris street; let through a
  dashcam picture with a smeared windshield and one handlebar picture.

Heavy blur can leave no edge for CPBD to measure; the blurred area then
decides alone.

### Excluding pictures that are the wrong way up

`predict` and `serve` exclude pictures whose content looks rotated (upside
down or sideways) with at least 99% confidence (`--min-orientation-confidence`;
`--no-orientation` skips it). See `panomoche/orientation.py`. A logistic
regression on frozen DINOv3 features learns which way is up from good pictures
rotated by 0/90/180/270°, so no labels are needed:

```bash
python scripts/orientation_eval.py     # trains models/orientation.joblib and evaluates it
```

* Rotated copies of 200 held-out pictures: 96.6% recognised.
* 1,300 real pictures: 5 predicted rotated with confidence 1.00, all really
  upside down or sideways (handlebar phone mounted the wrong way). Three of
  them had passed every other check. The 4 at 0.97-0.98 were upright: an
  aerial view, an information board, a view past a camper's ladder and a
  tilted fisheye road, hence the 0.99 minimum.
* It doesn't measure small tilts of the horizon yet.

**360° pictures.** A helmet camera mounted 90° off puts the helmet and the
ground in the middle of the picture and the sky on the sides. Cutting the
middle band into four crops, as before, gave these pictures only about 50%.
Now four ordinary 90° camera views are rendered from the whole sphere
(ahead, right, behind, left) and judged like flat pictures. A level camera
gives four upright views, a pitched one sideways views. A picture is
excluded when at least two views look rotated with 99% confidence. One
rotated view happens on clean pictures (a close wall, a facade seen from
below), and two views at 95% already caught a clean narrow-street picture.
On 491 360° pictures this catches 9 pictures of an LG helmet sequence, all
mounted 90° off, and no clean picture.

### Waviness and advertisement checks

Two standalone tools were merged in: the **Waviness Check** v0.2 and the
**Advertisement Check** v0.4. Their code is in `panomoche/waviness/` and
`panomoche/advertising/`, their reference pictures in `examples/`, and their
own tests run with the rest (`tests/test_waviness_tool.py`,
`tests/test_advertising_tool.py`). Their command lines and browser UIs are
kept:

```bash
panomoche waviness pictures/*.jpg --output-dir results    # JSON + overlay + edge map
panomoche waviness-ui --port 8765
panomoche ads pictures/*.jpg --output-dir ad-results
panomoche ads-ui --port 8766
```

`predict` and `serve` run both checks on every picture (`--no-waviness`,
`--no-ads` to skip them).

**Waviness** finds long, mostly vertical edges that bend together: the
rolling-shutter wobble of a vibrating mount (bent rails, rippling shopfronts).
On the 1,300 pictures it found 82 wavy flat pictures. 74 came from the Sony
handlebar sequence, where half the pictures are wavy and the wobble is real.
The 8 others were clean: a wide-angle church, a tunnel, foliage, Paris
streets. A lens, a curved building or tree trunks bend one picture's lines,
while a wobbling mount bends a long stretch of a sequence. So `wavy` only
excludes a picture when at least 30% of its neighbours are wavy too. Those 8
pictures were re-checked with their neighbours fetched from Panoramax: none
is excluded. A clean wide-angle Paris sequence had 4 wavy neighbours out of
20, so the limit sits above 20%. 46 Sony pictures are now excluded, all
visibly wobbly; the others were already excluded for blur. 360° pictures are
not checked: the projection curves straight edges, and every 360° "wavy"
result was a clean picture.

**Advertisement** (v0.4) rejects only obvious advertising graphics. Three
pieces of evidence are combined: a SigLIP 2 layout match, how much of the
picture is text, and commercial wording read by OCR (prices, "book now",
discounts, travel packages, contact details). OCR now runs on every
picture, about 0.5 s each. A picture is rejected only when the layout is
clear, there is substantial text, and there is either commercial wording or
a strongly poster-like layout. Anything ambiguous passes, and so does every
picture when the model or OCR is missing. On your 1,793 Panoramax pictures
nothing is rejected: the highest score is 79 against a threshold of 90, for
pictures with a lot of text such as signs and a price board. The tool's
`"` parsing bug with Tesseract 5.3 (see `classifier.py`) is fixed here. It
needs the model (`panomoche ads-setup`, or copy it to `models/siglip2/`) and
Tesseract (`sudo apt install tesseract-ocr`).

### Metadata checks

`fetch` and `predict` also use two values Panoramax stores with each picture
(see `panomoche/metadata.py`); a missing value never excludes a picture:

* **GPS accuracy** (`quality:horizontal_accuracy`) is recorded but doesn't
  exclude pictures by default. `--max-gps-accuracy 10` would exclude positions
  worse than 10 m. Most pictures report 2-5 m; in 700 random and Paris
  pictures, 9 reported 14-36 m, and accuracy was known for 61%.
* **Pixel density** (`panoramax:horizontal_pixel_density`): pictures below
  15 px per degree are excluded (`--min-pixel-density`). The limit stays under
  the 16 px/° of a normal GoPro Max 360° picture; 1 picture (11 px/°) was
  excluded. Density was known for 44%.

### Exposure: not a separate check (yet)

Brightness statistics were tested on 1,300 pictures and are not used. Real
exposure failures are rare (2-3 cases) and look like normal content to simple
statistics. A white Paris facade and a stone plaque came out "overexposed",
dark dashboards and dark foliage came out "underexposed", and the one real
night shot was missed because of its streetlights. TOPIQ already flags that
night shot as `low_quality`. A dependable check needs a model trained on
under- and overexposed examples (the DINOv3 classifier with the synthetic
exposure defects in `degrade.py`).

### Suggestions for Panoramax

* **Expose the blurred share.** The server already receives SGBlur's detections
  and stores only the road signs. Storing the aggregate share and count of
  blurred faces and plates as an item property (for example
  `panoramax:blurred_area`, next to the existing
  `panoramax:horizontal_pixel_density`) would reveal no positions, and would
  let clients filter without range requests on the original files.
* **Look at large low-confidence blurs.** Most of the blurred area comes from
  face detections below 25% confidence, often on facades. A size-dependent
  confidence threshold could keep privacy for real faces while hiding less of
  the street.

## Off-the-shelf models compared

`scripts/iqa_eval.py` scores a folder with three pretrained no-reference
models from [pyiqa](https://github.com/chaofengc/IQA-PyTorch): TOPIQ-NR and
MUSIQ (both trained on SPAQ) and LIQE-mix (which also gives a distortion type
and a scene type).

```bash
python scripts/iqa_eval.py data/eval --out data/iqa_eval.jsonl --html reports/iqa_eval.html
```

On 310 held-out Panoramax pictures (CPU, ~2 s per picture for all three):

* TOPIQ and MUSIQ agree well (Spearman 0.82); LIQE less so (0.65). TOPIQ is
  the one wired into `predict` (~0.4 s per view).
* They do find poor pictures: dashcam shots through a windshield with dashboard
  and reflections, a backlit underpass, soft or motion-blurred 360° pictures.
* They rank cameras more than pictures (see above).
* They don't know street-imagery defects: two dashcam pictures with the hood
  and wipers in the frame are among the 8 *best* rated.
* LIQE's distortion label is mostly `other` (62%) or `jpeg compression`, so it
  isn't very informative.

LIQE's first load builds a text-feature cache. pyiqa's stock code needs more
than 6 GB of RAM for it, so `IQAScorer` patches it to run in small batches.

## DINOv3 classifier (removed)

> This was the project's first approach. It is no longer in the code: TOPIQ and the rules
> above replaced it, and the `build`, `train` and `--scorer classifier` commands described
> below were removed in version 0.2. DINOv3 is still used for the orientation check and the
> rain experiment. The section is kept as a record of what was tried.


```
picture ──► prepare ──► views ──┬─► DINOv3 ViT-S/16 (frozen) ─► [CLS, mean patch] ─► mean+max over views ─┐
                                └─► pixel signals (sharpness, exposure, haze...) ─► mean/min/max ────────┤
                                                                                                         ▼
                                                     one logistic regression per issue  +  k-NN anomaly score
```

```bash
panomoche fetch --out data/raw                       # ~450 diverse pictures from 18 areas, ≤3 per sequence
panomoche build data/raw --out data/features.npz     # clean + 2 synthetic defects per picture
panomoche train --features data/features.npz --out models/quality.joblib --ablation
panomoche predict --scorer classifier --collection <sequence-uuid> --out seq.jsonl
```

* **Views.** A flat picture is one view (long side 512 px). A 360°
  equirectangular picture keeps only its horizon band (latitudes ±45°), split
  into four 90° views. The nadir and zenith are dropped on purpose, because a
  car roof, helmet or pole is always visible there on 360° rigs and must not
  count as an obstruction. Max-pooling over views keeps a defect that affects
  only one direction (for example a raindrop on one lens). The TOPIQ scorer
  uses the same views.
* **Why DINOv3 *and* pixel signals.** DINOv3 embeddings capture semantic
  problems well: something big in front of the lens, no street visible, drops
  on the lens. But DINO-style self-supervised training uses blur and
  colour-jitter augmentations, which teaches the embedding to partly *ignore*
  blur and exposure. A dozen cheap handcrafted statistics add that information
  back, and the classifier learns how to weigh both. `train --ablation`
  measures each feature set's contribution.
* **Training data without labels.** No labelled low-quality dataset exists for
  Panoramax yet. `build` takes real Panoramax pictures and synthesises each
  defect at a clearly unacceptable severity (see `panomoche/degrade.py`). When
  real reviewed labels become available, pass them with `--labels`. They are
  used as-is, alongside or instead of the synthetic samples.
* **Anomaly.** Each picture also gets its k-NN distance to known-good pictures
  in DINOv3 space. Pictures far from anything seen in training are flagged
  `atypical` (indoor shots, black frames, test pictures...).
* **Weights.** By default the backbone is `vit_small_patch16_dinov3.lvd1689m`
  from [timm](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m).
  timm redistributes the official DINOv3 checkpoints without the
  manual-approval gate of `facebook/dinov3-*`. Any timm DINOv3 ViT can be used
  with `build --backbone`.

### Improving it with real labels

The synthetic defects are a bootstrap. The main lever for real-world accuracy
is a few hundred reviewed real pictures:

1. `panomoche predict ... --html review.html`, then look at the top of the
   sheet (excluded first).
2. Write `labels.csv`:
   ```csv
   image,issues
   0a1b...jpg,blur
   3c4d...jpg,obstruction;underexposed
   5e6f...jpg,
   ```
   An empty `issues` field means the picture is fine. Real good pictures are
   as valuable as bad ones.
3. `panomoche build data/raw data/reviewed --labels labels.csv && panomoche train`.

## Rain on the lens (prototype)

Rain drops on the lens are local: a few soft, roughly round patches with a sharp scene around
them. The blur rule averages over the picture and passes such pictures: the rainy IGN GoPro Max
picture `581e9589` has 2.7% blurred area and a CPBD of 0.62 (very sharp edges), and TOPIQ 0.38
(above the 360° floor). Many drops also sit over sky or the car bonnet, which the blur measure
skips as plain.

**SigLIP 2 zero-shot** ("water droplets on the camera lens" against clear and blurry
descriptions) did not separate it from clean GoPro Max pictures: at 224 px the drops are a few
soft pixels.

**Block-sharpness rules on the sequence** were tried next, on the 30 frames around that picture
(the camera shoots every 2 s; IGN splits a drive into 2-picture sequences, so the frames were
found by place and time). Drops do stay at the same spot while the road and fields move behind
them, but the rules did not separate rainy from clean sequences:

* soft blocks staying in place while the content behind them changes: a second "clean" IGN
  sequence scored higher than the rainy one. It turned out to have water smears too (same
  rainy day), but clean GoPro Fusion, LG and dashcam sequences also scored like the rainy one;
* counting plain blocks as soft, since drops are so blurred that they lose medium-scale detail
  as well: the whole sky was marked, because clouds change between frames;
* requiring sharp detail around the patch: the drops of `581e9589` sit mostly over the bonnet
  and sky, so the rule scored it near zero while clean Fusion pictures scored higher.

**A DINOv3 classifier works** (`scripts/rain_model_eval.py`). Logistic regression on DINOv3
features (mean of the four 360° views), trained on the 16 frames of the smeared IGN sequence
and 60 clean GoPro Max pictures (same camera, so it cannot learn the camera). On the unseen
rainy sequence of `581e9589`: 30/30 frames detected (scores 0.75-0.93). On 30 unseen clean GoPro
Max pictures: 0 flagged (highest score 0.08), AUC 1.00.

Limits before it can exclude anything: both rainy sequences come from the same IGN survey on
the same rainy day, so the model may partly recognise wet grey weather rather than drops; it
has only seen GoPro Max 360° pictures. It needs rainy examples from other days, places and
cameras, and clean pictures under grey skies and on wet roads. It is not part of the decision.
