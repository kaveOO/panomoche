"""Loopback-only browser UI and JSON API of the advertisement check (``panomoche ads-ui``)."""
from __future__ import annotations

import argparse
import base64
import json
import re
import secrets
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .classifier import DEFAULT_THRESHOLD, VERSION, ContentFilter, failure_result, load_image, unchecked_result

ROOT = Path(__file__).resolve().parent / "web"                                 # index.html, app.js, style.css
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "advertising"    # reference images
MAX_REQUEST = 32 * 1024 * 1024
ENGINE = None
ANALYSIS_LOCK = threading.Lock()
EXPORT_LOCK = threading.Lock()
EXPORTS = OrderedDict()
EXAMPLES = [
    ("street.jpg", "Street.jpg"), ("landscape.jpg", "Mountain landscape.jpg"),
    ("advertisement.png", "Advertisement.png"),
    ("advertisement-with-street.png", "Advertisement with street.png"),
    ("screenshot.jpg", "Screenshot.jpg"), ("animals.png", "Animals.png"),
    ("document.png", "Document.png"), ("logo.png", "Logo.png"),
    ("blank.png", "Blank.png"), ("street-wavy.png", "Street with waviness.png"),
    ("street-with-sign.jpg", "Street with sale sign.jpg"),
    ("advertisement-panorama.png", "Panoramic advertisement.png"),
    ("park-statue.jpg", "Statue in a park.jpg"),
    ("ice-landscape.jpg", "Ice landscape.jpg"),
    ("advertisement-with-statue.png", "Advertisement with statue.png"),
    ("advertisement-with-ice.png", "Advertisement with ice landscape.png"),
    ("portrait.jpg", "Portrait.jpg"),
    ("advertisement-restaurant.png", "Restaurant advertisement.png"),
    ("advertisement-resort.png", "Resort advertisement.png")
]


class Handler(BaseHTTPRequestHandler):
    def send(self, status, payload, mime="application/json; charset=utf-8", filename=None):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def local_host(self):
        if self.headers.get("Host") not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
            self.send(403, {"error": "Use the local URL printed by server.py."})
            return False
        return True

    def do_GET(self):
        if not self.local_host():
            return
        path = urlparse(self.path).path
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        for file, _ in EXAMPLES:
            files[f"/examples/{file}"] = (EXAMPLES_DIR / file, "image/jpeg" if file.endswith(".jpg") else "image/png")
        if path == "/api/health":
            self.send(200, {"ready": ENGINE is not None, "version": VERSION, "model": "SigLIP 2", "ambiguous_action": "accept", "scope": "obvious_advertisements_only"})
        elif path == "/api/examples":
            self.send(200, [{"url": f"/examples/{file}", "name": name} for file, name in EXAMPLES if (EXAMPLES_DIR / file).is_file()])
        elif path.startswith("/api/download/"):
            with EXPORT_LOCK:
                export = EXPORTS.get(path.removeprefix("/api/download/"))
            if export and time.monotonic() - export[2] < 900:
                self.send(200, export[0], "application/json", export[1])
            else:
                self.send(410, {"error": "Download expired. Export again."})
        elif path in files:
            file, mime = files[path]
            target = ROOT / file if isinstance(file, str) else file
            self.send(200, target.read_bytes(), mime) if target.is_file() else self.send(404, {"error": "File not found."})
        else:
            self.send(404, {"error": "Page not found."})

    def do_POST(self):
        if not self.local_host():
            return
        path = urlparse(self.path).path
        if path not in {"/api/analyze", "/api/export"}:
            self.send(404, {"error": "Endpoint not found."})
            return
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            self.send(403, {"error": "Use the local tool tab."})
            return
        if self.headers.get_content_type() != "application/json":
            self.send(415, {"error": "Expected JSON."})
            return
        filename = "image"
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST:
                self.send(413, failure_result(filename, "Request must be between 1 byte and 32 MB."))
                return
            self.connection.settimeout(30)
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Invalid request.")
            filename = str(data.get("filename", "image"))[:200]
            if path == "/api/export":
                content = json.dumps(json.loads(data["content"]), indent=2, allow_nan=False).encode("utf-8")
                filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(data.get("filename", "content-report.json")))[:120]
                token = secrets.token_urlsafe(18)
                with EXPORT_LOCK:
                    EXPORTS[token] = (content, filename or "content-report.json", time.monotonic())
                    while len(EXPORTS) > 16:
                        EXPORTS.popitem(last=False)
                self.send(200, {"url": f"/api/download/{token}"})
                return
            # v0.2's `threshold` meant minimum place relevance. Ignore that
            # obsolete setting so old clients also get the relaxed policy.
            threshold = float(data.get("ad_threshold", DEFAULT_THRESHOLD))
            if not 90 <= threshold <= 99:
                raise ValueError("Advertisement rejection threshold must be between 90 and 99.")
            encoded = data.get("image")
            if not isinstance(encoded, str):
                raise ValueError("No image supplied.")
            image = load_image(base64.b64decode(encoded.split(",", 1)[-1], validate=True))
            if ENGINE is None:
                self.send(200, unchecked_result(filename, "The local visual model is unavailable.", image.size, threshold))
                return
            if not ANALYSIS_LOCK.acquire(blocking=False):
                self.send(429, failure_result(str(data.get("filename", "image")), "Another analysis is in progress. Try again."))
                return
            try:
                try:
                    result = ENGINE.analyze(image, threshold=threshold)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    result = unchecked_result(filename, "The local visual model failed.", image.size, threshold)
            finally:
                ANALYSIS_LOCK.release()
            result["filename"] = str(data.get("filename", "image"))[:200]
            self.send(200, result)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.send(400, {**failure_result(filename, error), "error": str(error)[:300]})
        except Exception:
            import traceback
            traceback.print_exc()
            self.send(500, failure_result(filename, "Request failed. No advertisement decision was made; check the local terminal."))


def main(argv=None):
    global ENGINE
    parser = argparse.ArgumentParser(prog="panomoche ads-ui", description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    print("Loading local advertisement model…", flush=True)
    try:
        ENGINE = ContentFilter()
    except Exception as error:
        print(f"Model unavailable: {error}. Valid images will pass unchecked.", flush=True)
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Content Check is ready at http://127.0.0.1:{server.server_port}", flush=True)
        print("Only obvious advertisements reject. Ambiguous content passes. Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
