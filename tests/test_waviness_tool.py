"""Waviness check (merged from the standalone tool's test_detector.py, unchanged checks).

Ground-truth synthetic controls, supplied examples, decoding and HTTP checks.
These checks do not estimate accuracy on a real camera dataset.
"""
import base64
import io
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image

from panomoche.waviness.detector import Settings, analyze, load_image
from panomoche.waviness.server import Handler

ROOT = Path(__file__).resolve().parents[1]   # examples/waviness/ holds the reference images


def facade():
    h, w = 720, 960
    image = np.full((h, w, 3), 225, np.uint8)
    for x in range(70, w - 30, 105):
        cv2.line(image, (x, 0), (x + 25, h - 1), (65, 65, 65), 3)
        cv2.line(image, (x + 13, 0), (x + 38, h - 1), (110, 110, 110), 2)
    for y in (95, 260, 435, 615):
        cv2.line(image, (0, y), (w - 1, y), (145, 145, 145), 3)
    return image


def warp(image, kind="sine", amount=12):
    h, w = image.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    if kind == "sine":
        displacement = amount * np.sin(2 * np.pi * y / 240)
    elif kind == "bow":
        displacement = amount * ((y - h / 2) / (h / 2)) ** 2
    else:
        raise ValueError(kind)
    return cv2.remap(image, x - displacement, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class DetectorTests(unittest.TestCase):
    def test_straight_facade(self):
        result, *_ = analyze(facade())
        self.assertEqual(result["label"], "no_strong_evidence")

    def test_blur_alone_does_not_flag(self):
        result, *_ = analyze(cv2.GaussianBlur(facade(), (0, 0), 3))
        self.assertEqual(result["label"], "no_strong_evidence")

    def test_smooth_bowing_does_not_flag(self):
        result, *_ = analyze(warp(facade(), "bow", 25))
        self.assertEqual(result["label"], "no_strong_evidence")

    def test_perspective_does_not_flag(self):
        matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [960, 0], [0, 720], [960, 720]]),
                                            np.float32([[110, 50], [860, 0], [20, 720], [960, 690]]))
        image = cv2.warpPerspective(facade(), matrix, (960, 720), borderValue=(225, 225, 225))
        result, *_ = analyze(image)
        self.assertEqual(result["label"], "no_strong_evidence")

    def test_shared_sine_wave_flags(self):
        for pixels in (8, 16):
            with self.subTest(displacement=pixels):
                result, *_ = analyze(warp(facade(), amount=pixels))
                self.assertEqual(result["label"], "wavy")
                self.assertGreaterEqual(result["independent_groups"], 3)

    def test_wave_survives_moderate_blur(self):
        image = cv2.GaussianBlur(warp(facade(), amount=16), (0, 0), 2)
        result, *_ = analyze(image)
        self.assertEqual(result["label"], "wavy")

    def test_sparse_scene_reports_insufficient_evidence(self):
        result, *_ = analyze(np.full((720, 960, 3), 180, np.uint8))
        self.assertEqual(result["label"], "insufficient_evidence")
        self.assertEqual(result["confidence"], "low")

    def test_texture_alone_does_not_flag(self):
        noise = np.random.default_rng(8).integers(0, 256, (720, 960, 3), dtype=np.uint8)
        result, *_ = analyze(noise)
        self.assertNotEqual(result["label"], "wavy")

    def test_unrelated_curves_do_not_flag(self):
        image = np.full((720, 960, 3), 225, np.uint8)
        y = np.arange(720)
        for i, x in enumerate(range(70, 950, 100)):
            displacement = 15 * np.sin(2 * np.pi * y / (150 + 40 * i) + i * 1.3)
            points = np.column_stack((x + displacement, y)).astype(np.int32)
            cv2.polylines(image, [points], False, (60, 60, 60), 3)
        result, *_ = analyze(image)
        self.assertNotEqual(result["label"], "wavy")

    def test_single_curved_structure_is_not_independent_support(self):
        image = np.full((720, 960, 3), 225, np.uint8)
        y = np.arange(720)
        x = 450 + 20 * np.sin(2 * np.pi * y / 240)
        cv2.polylines(image, [np.column_stack((x, y)).astype(np.int32)], False, (20, 20, 20), 8)
        result, *_ = analyze(image)
        self.assertNotEqual(result["label"], "wavy")

    def test_supplied_examples_flag(self):
        for number in (1, 2):
            with self.subTest(example=number):
                result, *_ = analyze(load_image((ROOT / "examples" / "waviness" / f"{number}.png").read_bytes()))
                self.assertEqual(result["label"], "wavy")
                self.assertFalse(result["score_is_probability"])
                json.dumps(result, allow_nan=False)

    def test_cows_are_not_rejected_as_wavy(self):
        image = load_image((ROOT / "examples" / "waviness" / "cows.png").read_bytes())
        result, *_ = analyze(image)
        self.assertEqual(result["label"], "insufficient_evidence")
        self.assertLess(result["score"], result["settings"]["decision_threshold"])
        self.assertEqual(result["confidence"], "low")

    def test_normal_street_is_not_rejected_as_wavy(self):
        image = load_image((ROOT / "examples" / "waviness" / "straight-street.jpg").read_bytes())
        for factor in (1, 2):
            with self.subTest(size_factor=factor):
                resized = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
                result, *_ = analyze(resized)
                self.assertEqual(result["label"], "no_strong_evidence")
                self.assertLess(result["score"], result["settings"]["decision_threshold"])
                self.assertLess(result["shared_edge_coverage_percent"], 10)

    def test_a_few_curves_cannot_outweigh_many_straight_edges(self):
        image = np.full((720, 960, 3), 225, np.uint8)
        y = np.arange(720)
        for i, x in enumerate(range(30, 940, 35)):
            displacement = 10 * np.sin(2 * np.pi * y / 240) if i in (2, 9, 16) else np.zeros_like(y)
            points = np.column_stack((x + displacement, y)).astype(np.int32)
            cv2.polylines(image, [points], False, (50, 50, 50), 2)
        result, *_ = analyze(image)
        self.assertLess(result["score"], result["settings"]["decision_threshold"])
        self.assertNotEqual(result["label"], "wavy")

    def test_reported_coverage_matches_edge_lengths(self):
        result, *_ = analyze(warp(facade()))
        all_length = sum(edge["span_pixels"] for edge in result["edges"])
        shared_length = sum(edge["span_pixels"] for edge in result["edges"] if edge["supported"])
        self.assertAlmostEqual(result["shared_edge_coverage_percent"],
                               shared_length / all_length * 100, delta=0.051)

    def test_invalid_image_and_settings(self):
        with self.assertRaises(ValueError):
            load_image(b"not an image")
        for settings in (Settings(decision_threshold=float("nan")), Settings(minimum_bending_percent=-1)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                settings.validate()

    def test_exif_orientation_is_applied(self):
        image = Image.new("RGB", (120, 180), "white")
        exif = Image.Exif()
        exif[274] = 6
        stream = io.BytesIO()
        image.save(stream, format="JPEG", exif=exif)
        self.assertEqual(load_image(stream.getvalue()).shape[:2], (120, 180))


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_upload_returns_result_and_png_overlays(self):
        ok, encoded = cv2.imencode(".png", warp(facade()))
        self.assertTrue(ok)
        payload = {"image": base64.b64encode(encoded).decode(), "filename": "synthetic.png"}
        request = Request(self.url + "/api/analyze", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=20) as response:
            result = json.load(response)
        self.assertEqual(result["label"], "wavy")
        for name in ("original", "overlay", "edge_map"):
            self.assertTrue(result[name].startswith("data:image/png;base64,"))
            pixels = cv2.imdecode(np.frombuffer(base64.b64decode(result[name].split(",", 1)[1]), np.uint8), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(pixels)

    def test_bad_upload_is_a_useful_error(self):
        request = Request(self.url + "/api/analyze", data=b'{"image":"invalid!!"}',
                          headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("error", json.load(caught.exception))

    def test_only_ui_files_are_served(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.url + "/detector.py", timeout=5)
        self.assertEqual(caught.exception.code, 404)

    def test_export_is_a_real_file_download(self):
        payload = {"filename": "result.json", "type": "application/json", "content": '{"score":42}'}
        request = Request(self.url + "/api/export", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            url = json.load(response)["url"]
        with urlopen(self.url + url, timeout=5) as response:
            self.assertEqual(response.headers["Content-Disposition"], 'attachment; filename="result.json"')
            self.assertEqual(json.load(response)["score"], 42)


if __name__ == "__main__":
    unittest.main()
