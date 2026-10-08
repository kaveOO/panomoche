# panomoche: architecture

A synthesis of the system, its API and its parts, with C4 diagrams (context, containers,
components) drawn as Mermaid flowcharts with the C4 notation and colours: they render on
GitHub and stay editable next to the code. How each rule was chosen is in
[METHODS.md](METHODS.md); how to run it is in the [README](../README.md).

## In one paragraph

panomoche sorts street-level pictures for Panoramax as **kept** or **excluded**, with one
reason. It reads pictures and their metadata from Panoramax instances, measures each picture
(metadata, privacy blur, orientation, sharpness, advertisement, waviness, image quality),
and applies one decision that judges each picture against its own sequence. It runs as a
command-line tool for batches and as a local web app with an HTTP API, which a Panoramax
server could call on upload, the way it already calls SGBlur to blur faces and plates.
Everything runs on CPU, locally; nothing is written to Panoramax unless asked explicitly.

## Level 1: system context

```mermaid
flowchart LR
    reviewer["<b>Reviewer</b><br/>[Person]<br/>Reviews verdicts,<br/>tests uploads"]
    operator["<b>Operator</b><br/>[Person]<br/>Runs batches:<br/>fetch, predict, tag"]
    panomoche["<b>panomoche</b><br/>[Software system]<br/>Measures street-level pictures and<br/>sorts them as kept or excluded,<br/>with a reason"]
    panoramax["<b>Panoramax instances</b><br/>[External system]<br/>STAC API, picture files,<br/>semantics tags, reports"]
    sgblur["<b>SGBlur</b><br/>[External system]<br/>Face and plate blurring<br/>on upload"]
    hubs["<b>Model sources</b><br/>[External system]<br/>Hugging Face, timm, pyiqa"]
    reviewer -- "Reviews, uploads<br/>[browser, HTTP]" --> panomoche
    operator -- "fetch, predict, tag<br/>[CLI]" --> panomoche
    panomoche -- "Searches and downloads pictures;<br/>writes tags (dry run unless --apply)<br/>[HTTPS, STAC]" --> panoramax
    panoramax -- "Sends each upload;<br/>gets it back blurred,<br/>with a record" --> sgblur
    panomoche -- "Model weights, once<br/>[HTTPS]" --> hubs
    panoramax -. "Could call on upload:<br/>POST /predict (future)" .-> panomoche
    classDef person fill:#08427b,stroke:#052e56,color:#fff
    classDef system fill:#1168bd,stroke:#0b4884,color:#fff
    classDef container fill:#438dd5,stroke:#2e6295,color:#fff
    classDef component fill:#85bbf0,stroke:#5d82a8,color:#000
    classDef store fill:#438dd5,stroke:#2e6295,color:#fff
    classDef ext fill:#999999,stroke:#6b6b6b,color:#fff
    class reviewer,operator person
    class panomoche system
    class panoramax,sgblur,hubs ext
```

## Level 2: containers

```mermaid
flowchart TB
    reviewer["<b>Reviewer</b><br/>[Person]"]
    operator["<b>Operator</b><br/>[Person]"]
    panoramax["<b>Panoramax instances</b><br/>[External system]<br/>STAC API, picture files"]
    subgraph boundary["panomoche: one Python package, runs locally on CPU"]
        direction TB
        pages["<b>Pages</b><br/>[Container: HTML, CSS, JavaScript]<br/>Upload page, review page"]
        cli["<b>Command line</b><br/>[Container: Python]<br/>fetch, predict, tag, serve;<br/>merged tools: waviness, ads"]
        web["<b>Web app and HTTP API</b><br/>[Container: FastAPI, Uvicorn]<br/>/predict, /api/sequence"]
        engine["<b>Judging engine</b><br/>[Container: PyTorch, OpenCV, ONNX Runtime]<br/>Measures every picture,<br/>applies the one decision"]
        data[("<b>Datasets</b><br/>[JSONL, JPEG]<br/>pictures, predictions")]
        models[("<b>Model files</b><br/>[PyTorch, ONNX, joblib]<br/>TOPIQ, DINOv3, SigLIP 2,<br/>orientation classifier")]
    end
    reviewer -- "Uses [browser]" --> pages
    operator -- "Runs" --> cli
    pages -- "JSON over HTTP" --> web
    cli -- "Starts (serve)" --> web
    web -- "Measures and decides" --> engine
    cli -- "Measures and decides" --> engine
    cli -- "Searches, downloads, tags<br/>[HTTPS]" --> panoramax
    cli -- "Reads, writes" --> data
    web -- "Reads predictions" --> data
    engine -- "Loads" --> models
    classDef person fill:#08427b,stroke:#052e56,color:#fff
    classDef system fill:#1168bd,stroke:#0b4884,color:#fff
    classDef container fill:#438dd5,stroke:#2e6295,color:#fff
    classDef component fill:#85bbf0,stroke:#5d82a8,color:#000
    classDef store fill:#438dd5,stroke:#2e6295,color:#fff
    classDef ext fill:#999999,stroke:#6b6b6b,color:#fff
    class reviewer,operator person
    class pages,cli,web,engine container
    class data,models store
    class panoramax ext
    style boundary fill:none,stroke:#444,stroke-dasharray:6 4
```

## Level 3: components of the judging engine

```mermaid
flowchart TB
    picture["Picture + Panoramax metadata<br/>(+ its sequence neighbours)"]
    subgraph engine["Judging engine"]
        imaging["<b>imaging</b> [Pillow, NumPy]<br/>Loads, detects 360°, cuts views"]
        metadata["<b>metadata</b><br/>px per degree"]
        privacy["<b>privacy</b><br/>SGBlur record<br/>(HTTP Range)"]
        orientation["<b>orientation</b><br/>DINOv3 +<br/>logistic regression"]
        sharpness["<b>sharpness</b><br/>blurred area<br/>+ CPBD (OpenCV)"]
        advertising["<b>advertising</b><br/>SigLIP 2 +<br/>Tesseract OCR"]
        waviness["<b>waviness</b><br/>edge bending<br/>(OpenCV)"]
        iqa["<b>iqa / scoring</b><br/>TOPIQ-NR<br/>(pyiqa, PyTorch)"]
        decision["<b>decision</b> [Python]<br/>one set of rules, with the sequence neighbours"]
    end
    verdict["<b>Excluded</b> with one reason, or <b>kept</b>"]
    picture --> imaging
    imaging --> metadata & privacy & orientation & sharpness & advertising & waviness & iqa
    metadata -- "low_resolution" --> decision
    privacy -- "privacy_blur" --> decision
    orientation -- "wrong_orientation" --> decision
    sharpness -- "not_sharp" --> decision
    advertising -- "advertisement" --> decision
    waviness -- "wavy" --> decision
    iqa -- "low_quality,<br/>sequence_outlier" --> decision
    decision --> verdict
    classDef component fill:#85bbf0,stroke:#5d82a8,color:#000
    classDef io fill:#ffffff,stroke:#444,color:#000
    class imaging,metadata,privacy,orientation,sharpness,advertising,waviness,iqa,decision component
    class picture,verdict io
    style engine fill:none,stroke:#444,stroke-dasharray:6 4
```

Supporting modules: `panoramax.py` (API client, sampling, semantics payloads), `dedupe.py`
(known pictures, duplicates), `report.py` (review page rendering), `sequence.py`
(per-sequence summaries), `backbone.py` (DINOv3 feature extractor).

## How one upload is judged

```mermaid
sequenceDiagram
    autonumber
    actor R as Reviewer or Panoramax server
    participant W as Web app (/predict)
    participant M as Measurements
    participant D as decision
    R->>W: POST /predict (picture, optional SGBlur record)
    W->>W: known picture of the review page? reuse its metadata, privacy record and sequence
    W->>M: privacy blur, sharpness, orientation, waviness, advertisement, TOPIQ
    M-->>W: measurements
    W->>D: decide(picture, with its sequence neighbours if known)
    D-->>W: excluded (reason) or kept
    W-->>R: verdict, measurements, Panoramax tags it would write
```

## Decision rules

A picture is excluded, with the first reason that applies; otherwise it is kept.

| Reason | Rule |
|---|---|
| `low_resolution` | under 15 px per degree |
| `advertisement` | layout, text area and commercial wording all point to an advertising graphic |
| `privacy_blur` | SGBlur's blurred boxes cover more than 5% of the picture |
| `wrong_orientation` | rotated with ≥ 99% confidence; 360°: 2 of 4 rendered views |
| `not_sharp` | ≥ 33% of the textured area blurred with CPBD < 0.15, or ≥ 20% with CPBD < 0.12 |
| `wavy` | rolling-shutter wobble confirmed by ≥ 30% of the sequence neighbours (flat pictures) |
| `sequence_outlier` | TOPIQ ≥ 0.12 below the median of the 10 pictures before and after |
| `low_quality` | TOPIQ < 0.32 (360° or with neighbours), < 0.40 (flat picture alone) |

## HTTP API

Served by `panomoche serve` on `127.0.0.1:8000`.

| Method and path | Input | Output |
|---|---|---|
| `GET /` | | upload page |
| `GET /review` | | review page of the predictions file (`--predictions`) |
| `GET /images/{id}?size=thumb\|full` | picture id from the predictions file | JPEG |
| `GET /api/info` | | model, thresholds, which checks are on, number of predictions |
| `POST /predict` | multipart `picture`; optional `is_pano`, `sgblur` (SGBlur's `x-sgblur` header), `tag_key` | verdict (`excluded` reason or null), every measurement, Panoramax semantics tags |
| `POST /api/sequence` | JSON `{"pictures": [measured pictures, in order]}` | each picture re-decided as one sequence |

The merged tools keep their own small servers (`panomoche waviness-ui`, `panomoche ads-ui`) with
`POST /api/analyze`, `POST /api/export` and `GET /api/health`.

## Panoramax interface

| Call | Use |
|---|---|
| `GET /api/search` (STAC, bbox, datetime) | sampling, finding neighbours |
| `GET /api/collections/{id}/items` | a whole sequence, in capture order |
| picture assets (`sd`, `hd`) | `sd` (2048 px) for measuring; the end of `hd` (HTTP Range) for SGBlur's record |
| `PATCH /api/collections/{cid}/items/{id}` (`semantics`) | `panomoche tag --apply`: tags such as `quality_issue=sequence_outlier` with `detection_model[...]` |
| `POST /api/reports` | `tag --apply --report`: `picture_low_quality` moderation reports |

## Data

| File | Content |
|---|---|
| `data/<set>/` + `meta.jsonl` | downloaded pictures and their Panoramax metadata (`fetch`) |
| `data/<set>.jsonl` | one prediction per picture: measurements, `excluded`, `context` (`predict`) |
| `models/` | `orientation.joblib`, `siglip2/` (TOPIQ and DINOv3 weights are cached by pyiqa and timm) |

## Quality and limits

* 169 tests (pytest); lint and format with ruff; dead code checked with vulture.
* Benchmarks: blur detection catches 100% of visible focus blur and 56-74% of motion blur at a
  3.5% false-alarm rate ([reports/blur_benchmark.md](../reports/blur_benchmark.md)).
* Known limits: motion smear, rain on the lens (prototype only), sequences with a uniform fault,
  small horizon tilts. See the README, "Things to know".
