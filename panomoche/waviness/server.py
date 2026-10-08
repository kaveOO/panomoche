"""Local browser UI of the waviness check (``panomoche waviness-ui``). No images leave this computer."""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
import secrets
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2

from .detector import Settings, analyze, encode_png, load_image

ROOT = Path(__file__).resolve().parent / "web"                                 # index.html, app.js, style.css
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "waviness"       # reference images


def asset(filename: str) -> Path:
    return EXAMPLES / filename.removeprefix("examples/") if filename.startswith("examples/") else ROOT / filename
MAX_REQUEST = 32 * 1024 * 1024
ANALYSIS_SLOTS = threading.BoundedSemaphore(2)
EXPORTS = OrderedDict()
EXPORT_LOCK = threading.Lock()
cv2.setNumThreads(2)


class Handler(BaseHTTPRequestHandler):
    def send(self, status, payload, content_type="application/json; charset=utf-8", filename=None):
        if not isinstance(payload, bytes):
            payload = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.headers.get("Host") not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
            self.send(403, {"error": "Use the local address printed by server.py."})
            return
        path = urlparse(self.path).path
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8"),
                 "/examples/1.png": ("examples/1.png", "image/png"),
                 "/examples/2.png": ("examples/2.png", "image/png"),
                 "/examples/cows.png": ("examples/cows.png", "image/png"),
                 "/examples/straight-street.jpg": ("examples/straight-street.jpg", "image/jpeg")}
        if path.startswith("/api/download/"):
            token = path.removeprefix("/api/download/")
            with EXPORT_LOCK:
                item = EXPORTS.get(token)
            if item and time.monotonic() - item[3] < 900:
                content, mime, name, _ = item
                self.send(200, content, mime, name)
            else:
                self.send(410, {"error": "This download expired. Export the result again."})
        elif path == "/api/health":
            self.send(200, {"ok": True})
        elif path in files:
            filename, mime = files[path]
            file = asset(filename)
            if file.is_file():
                self.send(200, file.read_bytes(), mime)
            else:
                self.send(404, {"error": "File not found."})
        else:
            self.send(404, {"error": "Page not found."})

    def do_POST(self):
        path = urlparse(self.path).path
        if self.headers.get("Host") not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
            self.send(403, {"error": "Use the local address printed by server.py."})
            return
        if path not in {"/api/analyze", "/api/export"}:
            self.send(404, {"error": "Endpoint not found."})
            return
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            self.send(403, {"error": "Open the tool directly in its local browser tab."})
            return
        if self.headers.get_content_type() != "application/json":
            self.send(415, {"error": "Expected a JSON image request."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send(400, {"error": "Invalid request length."})
            return
        if length <= 0 or length > MAX_REQUEST:
            self.send(413, {"error": "Image request exceeds 32 MB. Use a smaller image."})
            return
        self.connection.settimeout(30)
        if path == "/api/export":
            try:
                payload = json.loads(self.rfile.read(length))
                name = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(payload.get("filename", "result")))[:120]
                mime = payload.get("type")
                if mime == "application/json":
                    content = json.dumps(json.loads(payload["content"]), indent=2, allow_nan=False).encode("utf-8")
                elif mime == "image/png":
                    content = base64.b64decode(payload["content"].split(",", 1)[-1], validate=True)
                    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise ValueError("Invalid overlay data.")
                else:
                    raise ValueError("Unsupported export format.")
                token = secrets.token_urlsafe(18)
                with EXPORT_LOCK:
                    EXPORTS[token] = (content, mime, name, time.monotonic())
                    while len(EXPORTS) > 16:
                        EXPORTS.popitem(last=False)
                self.send(200, {"url": f"/api/download/{token}"})
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                self.send(400, {"error": str(error)[:200]})
            return
        if not ANALYSIS_SLOTS.acquire(blocking=False):
            self.send(429, {"error": "Two images are already being analyzed. Try again shortly."})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not isinstance(payload.get("image"), str):
                raise ValueError("No image data was supplied.")
            encoded = payload["image"]
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[-1]
            data = base64.b64decode(encoded, validate=True)
            supplied = payload.get("settings", {})
            if not isinstance(supplied, dict):
                raise ValueError("Invalid settings.")
            # Only expose settings meaningful in the UI. Defaults are stable
            # and every analysis records the settings used.
            allowed = {"minimum_bending_percent", "decision_threshold"}
            settings = Settings(**{key: float(value) for key, value in supplied.items() if key in allowed})
            result, original, overlay, edges = analyze(load_image(data), settings)
            result["filename"] = str(payload.get("filename", "image"))[:200]
            for name, image in (("original", original), ("overlay", overlay), ("edge_map", edges)):
                result[name] = "data:image/png;base64," + base64.b64encode(encode_png(image)).decode("ascii")
            self.send(200, result)
        except (ValueError, TypeError, binascii.Error, cv2.error) as error:
            self.send(400, {"error": str(error)[:300]})
        except Exception:
            self.send(500, {"error": "Analysis failed. Check the local terminal for details."})
            import traceback
            traceback.print_exc()
        finally:
            ANALYSIS_SLOTS.release()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="panomoche waviness-ui", description="Run the local image waviness tool.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Waviness Check is ready at http://127.0.0.1:{server.server_port}", flush=True)
        print("Images stay on this computer. Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
