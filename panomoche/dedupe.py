"""Keep a dataset free of repeats: pictures already seen, and duplicate or near-identical images.

* Known pictures: every id already in a predictions/metadata file or in the reviewer's labels is
  skipped when fetching, so a new sample never brings back a picture already judged.
* Duplicates: the same file (SHA-1), or near-identical content (difference hash, 64 bits, at most
  ``MAX_DISTANCE`` bits apart), e.g. a camera that kept shooting while stopped at a red light,
  or the same picture uploaded twice. The first one met is kept.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

MAX_DISTANCE = 6  # of 64 bits; consecutive frames of a moving camera differ by 15 or more


def known_ids(paths) -> set[str]:
    """Picture ids listed in JSONL files (``{"id": ...}`` per line) or in a labels JSON (id -> label)."""
    ids: set[str] = set()
    for path in map(Path, paths):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            ids.update(json.loads(path.read_text()))
        else:
            for line in path.read_text().splitlines():
                if line.strip():
                    ids.add(json.loads(line)["id"])
    return ids


def known_collections(paths) -> set[str]:
    out: set[str] = set()
    for path in map(Path, paths):
        if path.is_file() and path.suffix == ".jsonl":
            out.update(
                r["collection"]
                for r in map(json.loads, filter(str.strip, path.read_text().splitlines()))
                if r.get("collection")
            )
    return out


def dhash(path) -> np.ndarray:
    """64-bit difference hash: does each pixel of a 9x8 greyscale thumbnail get brighter to the right?"""
    a = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS), np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def sha1(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def drop_duplicates(records: list[dict], reference: list[dict] = (), max_distance: int = MAX_DISTANCE):
    """Split ``records`` (each with a ``path``) into (kept, duplicates); ``reference`` pictures count as seen.

    Each duplicate gets ``duplicate_of``: the id of the picture it repeats.
    """
    seen_sha: dict[str, str] = {}
    seen_hash: list[tuple[np.ndarray, str]] = []
    for r in reference:
        if Path(r["path"]).is_file():
            seen_sha[sha1(r["path"])] = r["id"]
            seen_hash.append((dhash(r["path"]), r["id"]))
    kept, dups = [], []
    for r in records:
        digest, h = sha1(r["path"]), dhash(r["path"])
        same = seen_sha.get(digest) or next((i for g, i in seen_hash if int((g != h).sum()) <= max_distance), None)
        if same:
            dups.append({**r, "duplicate_of": same})
            continue
        seen_sha[digest] = r["id"]
        seen_hash.append((h, r["id"]))
        kept.append(r)
    return kept, dups
