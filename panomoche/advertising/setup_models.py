"""Download pinned model data, verify SHA-256 and prepare text embeddings.

No uploaded images are transmitted. This is the only network-enabled setup step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .classifier import DEFAULT_MODEL_DIR

REPOSITORY = "onnx-community/siglip2-base-patch16-224-ONNX"
REVISION = "ba1f3b0843f24bc5417d38e19c37b287d719b2f4"
FILES = {
    "onnx/vision_model_quantized.onnx": "5f2b401c1a4fc095702a5d45348e17ad46c4f87064085365b43c6e8eaa5c0070",
    "onnx/text_model_quantized.onnx": "3a0603d3a00c05a80a6ded4743c16aaac7b1e62cdcc7e362e7ce418659b96400",
    "tokenizer.json": "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"
}

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def download_one(model_dir, filename, expected):
    target = model_dir / Path(filename).name
    if target.is_file() and digest(target) == expected:
        print(f"Verified {target.name}", flush=True)
        return
    temporary = target.with_suffix(target.suffix + ".partial")
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{filename}?download=true"
    print(f"Downloading {target.name}…", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Panoramax-Content-Filter/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            output.write(block)
    if digest(temporary) != expected:
        temporary.unlink()
        raise ValueError(f"Checksum verification failed for {target.name}.")
    temporary.replace(target)
    print(f"Ready: {target.name}", flush=True)

def main(argv=None):
    parser = argparse.ArgumentParser(prog="panomoche ads-setup", description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args(argv)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(download_one, args.model_dir, filename, expected) for filename, expected in FILES.items()]
        for future in futures:
            future.result()
    (args.model_dir / "manifest.json").write_text(json.dumps({"repository": REPOSITORY, "revision": REVISION, "sha256": FILES}, indent=2))
    if not args.download_only:
        os.environ["CONTENT_MODEL_DIR"] = str(args.model_dir.resolve())
        from .classifier import prepare_text_embeddings
        prepare_text_embeddings(args.model_dir)
    print("Model setup complete.", flush=True)

if __name__ == "__main__":
    main()
