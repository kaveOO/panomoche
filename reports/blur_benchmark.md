# Blur detection benchmark

What is measured: the `not_sharp` rule (blurred area + CPBD, `panomoche/sharpness.py`) and
four alternatives, on 200 random pictures of the new dataset (never used to tune the rule),
each untouched and blurred 6 ways at viewing size (1024 px), JPEG-encoded like Panoramax files.
Each alternative is set to flag as many untouched pictures as the rule (3.5%), so the columns
compare how much blur each one catches at the same false-alarm rate.
Re-run: `python scripts/blur_benchmark.py --n 200` (about 20 minutes).

Pictures: 200 (45 360°, 50 cameras) from data/new.jsonl, each untouched and blurred 6 ways. Untouched pictures flagged by the rule: 3.5% (the other methods are set to the same rate).

| blur | rule | blurred area alone | CPBD alone | Laplacian variance | TOPIQ |
|---|---|---|---|---|---|
| original | 4% | 4% | 4% | 4% | 4% |
| gaussian σ 0.75 px | 50% | 34% | 86% | 32% | 14% |
| gaussian σ 1.5 px | 100% | 100% | 100% | 100% | 36% |
| gaussian σ 3 px | 100% | 100% | 100% | 100% | 95% |
| motion 5 px | 56% | 48% | 74% | 36% | 28% |
| motion 11 px | 74% | 64% | 87% | 78% | 85% |
| motion 21 px | 73% | 64% | 94% | 91% | 100% |

Median time per picture: blur rule (area + CPBD) 23.4 ms, Laplacian variance 1.8 ms, TOPIQ 221.9 ms


## What it shows

* **Focus blur:** σ 1.5 px and more is always caught. Very light blur (σ 0.75 px, barely
  visible at viewing size) is caught half the time. TOPIQ is far weaker (36% at σ 1.5).
* **Motion blur is the weak spot:** 56% / 74% / 73% for 5 / 11 / 21 px. Even strong motion
  blur escapes on about 1 flat picture in 3. On those pictures CPBD is near zero (few sharp
  edges), but the blurred area is only about 11%. Motion blur keeps fine detail across the
  motion direction, so the area measure underrates it; this is also why the LG 360° smear
  escapes. CPBD alone catches 87–94% of the motion blur.
* **Why not CPBD alone:** on real pictures it calls soft but clean foliage, churches and
  small phone-app pictures blurred (see docs/METHODS.md, "Excluding pictures that aren't sharp"). A
  third condition (blurred area ≥ 8% and CPBD < 0.06) raises strong motion blur from 73% to
  84% with no extra false alarm here. On the 1,793 real pictures it would exclude 3 more:
  one plausibly blurred dashcam strip and two soft but clean landscape pictures. Not adopted.
* **Untouched pictures flagged: 3.5%** (7 of 200), the same pictures `predict` excludes as
  not sharp. Checked by eye on the whole new dataset: mostly rain, night, dashboards and
  side-window motion, plus a few borderline soft phone pictures.
* **Speed:** the rule takes 23 ms per picture, a hundredth of the time of the whole
  `predict` pipeline.
