"""Panoramax API client: fetch pictures, and write quality results back.

Reads go through any STAC endpoint (by default the federated meta-catalog
``api.panoramax.xyz``). Writes must go to the instance that *hosts* the
picture (IGN, OSM-FR, ...), which the meta-catalog exposes as the ``via`` link.

Two write paths exist in the GeoVisio API (v2.16):

* semantics tags: ``PATCH /api/collections/{cid}/items/{id}`` with
  ``{"semantics": [{"key", "value", "action"}]}`` — anyone may add tags;
* moderation reports: ``POST /api/reports`` with ``issue=picture_low_quality``.

Tags follow the Panoramax qualifier convention already used by detection bots,
e.g. ``detection_model[osm|traffic_sign=yes]=SGBlur-yolo11s/0.1.0``. There is no
standard quality tag yet, so the key is configurable and defaults to
``quality_issue`` (no prefix = Panoramax-own tag); it should be agreed with the
community before writing at scale.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import requests

from . import __version__

DEFAULT_API = "https://api.panoramax.xyz/api"
DEFAULT_TAG_KEY = "quality_issue"
FRANCE = (-5.2, 42.3, 8.3, 51.1)  # metropolitan France, where most Panoramax pictures are
USER_AGENT = f"panomoche/{__version__} (low-quality picture detection research)"


def item_is_pano(item: dict) -> bool | None:
    fov = (item.get("properties", {}).get("pers:interior_orientation") or {}).get("field_of_view")
    return None if fov is None else fov >= 360


def item_instance_api(item: dict) -> str | None:
    """Root API URL of the instance hosting this picture."""
    for link in item.get("links", []):
        if link.get("rel") == "via":
            return link["href"].rstrip("/") + "/api"
    for link in item.get("links", []):
        if link.get("rel") == "root":
            return link["href"].rstrip("/")
    return None


def item_asset_url(item: dict, asset: str = "hd") -> str | None:
    """Absolute URL of an asset; some instances publish hrefs relative to their own host."""
    href = item.get("assets", {}).get(asset, {}).get("href")
    if href and href.startswith("/"):
        api = item_instance_api(item) or ""
        href = api.rsplit("/api", 1)[0] + href
    return href


def item_record(item: dict) -> dict:
    """Small metadata record kept next to downloaded pictures."""
    p = item.get("properties", {})
    io = p.get("pers:interior_orientation") or {}
    return {
        "id": item["id"],
        "collection": item.get("collection"),
        "is_pano": item_is_pano(item),
        "camera": " ".join(x for x in (io.get("camera_manufacturer"), io.get("camera_model")) if x) or None,
        # size of the original file; the "sd" asset is always 2048 px wide, so smaller originals are enlarged
        "original_size": io.get("sensor_array_dimensions"),
        "datetime": p.get("datetime"),
        "rank": p.get("geovisio:rank_in_collection"),
        "gps_accuracy": p.get("quality:horizontal_accuracy"),  # metres
        "pixel_density": p.get("panoramax:horizontal_pixel_density"),  # px per degree
        "license": p.get("license"),
        "instance_api": item_instance_api(item),
        "hd_url": item_asset_url(item, "hd"),  # carries SGBlur's record of what it blurred
        "lon_lat": item.get("geometry", {}).get("coordinates"),
    }


def semantics_for(prediction: dict, tag_key: str = DEFAULT_TAG_KEY, action: str = "add") -> list[dict]:
    """Semantics tags describing the issues of one prediction."""
    tags = []
    for issue in prediction["issues"]:
        target = f"{tag_key}={issue}"
        tags += [
            {"key": tag_key, "value": issue, "action": action},
            {"key": f"detection_model[{target}]", "value": prediction["model"], "action": action},
        ]
    return tags


class PanoramaxClient:
    def __init__(self, api_url: str = DEFAULT_API, token: str | None = None, delay: float = 0.2):
        self.api_url = api_url.rstrip("/")
        self.delay = delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def _get(self, url: str, **params) -> dict:
        r = self.session.get(url, params=params or None, timeout=60)
        r.raise_for_status()
        time.sleep(self.delay)
        return r.json()

    def search(self, bbox: Iterable[float] | None = None, limit: int = 100, **filters) -> list[dict]:
        params = {"limit": limit, **filters}
        if bbox is not None:
            params["bbox"] = ",".join(str(v) for v in bbox)
        return self._get(f"{self.api_url}/search", **params).get("features", [])

    def collection_items(
        self, collection_id: str, limit: int | None = None, start_after_rank: int | None = None
    ) -> Iterator[dict]:
        url, params, n = f"{self.api_url}/collections/{collection_id}/items", {"limit": 100}, 0
        if start_after_rank:
            params["startAfterRank"] = start_after_rank
        while url:
            page = self._get(url, **params)
            for item in page.get("features", []):
                rank = item.get("properties", {}).get("geovisio:rank_in_collection")
                if n == 0 and start_after_rank and rank is not None and rank <= start_after_rank:
                    # The federated meta-catalog silently ignores startAfterRank (checked Oct 2026).
                    raise ValueError(
                        f"{self.api_url} ignores startAfterRank; query the instance hosting the "
                        "sequence instead (e.g. --api https://panoramax.openstreetmap.fr/api)"
                    )
                yield item
                n += 1
                if limit and n >= limit:
                    return
            url, params = next((link["href"] for link in page.get("links", []) if link.get("rel") == "next"), None), {}

    def item(self, collection_id: str, item_id: str) -> dict:
        return self._get(f"{self.api_url}/collections/{collection_id}/items/{item_id}")

    def download(self, item: dict, dest_dir: Path, asset: str = "sd") -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{item['id']}.jpg"
        if path.exists() and path.stat().st_size > 0:
            return path
        r = self.session.get(item["assets"][asset]["href"], timeout=120)
        r.raise_for_status()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.rename(path)
        time.sleep(self.delay)
        return path

    def sample(self, bboxes: list[list[float]], per_bbox: int, max_per_collection: int, seed: int = 0) -> list[dict]:
        """Diverse sample: consecutive pictures of a sequence are near-duplicates, so cap per collection."""
        rng = random.Random(seed)
        picked: dict[str, dict] = {}
        for bbox in bboxes:
            by_col: dict[str, list[dict]] = defaultdict(list)
            for item in self.search(bbox=bbox, limit=500):
                by_col[item.get("collection")].append(item)
            chosen: list[dict] = []
            for items in by_col.values():
                chosen += rng.sample(items, min(max_per_collection, len(items)))
            rng.shuffle(chosen)
            for item in chosen[:per_bbox]:
                picked.setdefault(item["id"], item)
        return list(picked.values())

    def random_sample(
        self,
        n: int,
        region=FRANCE,
        size: float = 0.2,
        seed: int = 0,
        max_searches: int | None = None,
        skip_ids=(),
        skip_collections=(),
    ) -> list[dict]:
        """``n`` pictures from random places, at most one per sequence.

        Each draw searches a random ``size``° square in ``region`` and keeps one
        random picture of a sequence not seen yet, so the sample spreads over
        places, contributors and cameras instead of following a few sequences.
        Squares without pictures (sea, countryside without coverage) are skipped, and so are
        ``skip_ids`` (pictures already judged) and ``skip_collections`` (sequences already sampled).
        """
        rng = random.Random(seed)
        picked: list[dict] = []
        seen: set[str] = set(skip_collections)
        skip_ids = set(skip_ids)
        for _ in range(max_searches or n * 20):
            if len(picked) >= n:
                break
            lon = rng.uniform(region[0], region[2] - size)
            lat = rng.uniform(region[1], region[3] - size)
            by_col: dict[str, list[dict]] = defaultdict(list)
            for item in self.search(bbox=[lon, lat, lon + size, lat + size], limit=100):
                if item.get("collection") not in seen and item.get("id") not in skip_ids:
                    by_col[item.get("collection")].append(item)
            if by_col:
                collection = rng.choice(sorted(by_col, key=str))
                picked.append(rng.choice(by_col[collection]))
                seen.add(collection)
        return picked

    # --- writes (instance API, authenticated) -------------------------------------------------

    def add_semantics(self, instance_api: str, collection_id: str, item_id: str, tags: list[dict]) -> dict:
        r = self.session.patch(
            f"{instance_api}/collections/{collection_id}/items/{item_id}", json={"semantics": tags}, timeout=60
        )
        r.raise_for_status()
        return r.json()

    def report(self, instance_api: str, item_id: str, comment: str) -> dict:
        r = self.session.post(
            f"{instance_api}/reports",
            timeout=60,
            json={"issue": "picture_low_quality", "picture_id": item_id, "reporter_comments": comment},
        )
        r.raise_for_status()
        return r.json()


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
