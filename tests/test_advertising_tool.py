"""Advertisement check (merged from the standalone tool v0.4's test_filter.py, checks unchanged).

Regression checks for an advertisement-only, accept-by-default policy.

Examples are development checks, not a representative accuracy dataset.
Run after model setup: python -m unittest -v test_filter
"""
import base64
import io
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from panomoche.advertising import server
from panomoche.advertising.classifier import ContentFilter, commercial_evidence, decide, failure_result, load_image

ROOT = Path(__file__).resolve().parents[1] / "examples" / "advertising"
def text_evidence(text="", area=0.0, words=12):
    commerce = commercial_evidence(text)
    return {"available": True, "text_area_percent": area, "words": words,
            "commercial_strength": commerce["strength"]}


class DecisionTests(unittest.TestCase):
    def test_ordinary_reference_accepts(self):
        self.assertEqual(decide(0.15, 0.09, ocr=text_evidence())[0], "accept")

    def test_visual_evidence_is_not_erased_when_ocr_is_missing(self):
        decision, score, reason = decide(0.05, 0.20)
        self.assertEqual(decision, "accept")
        self.assertGreater(score, 90)
        self.assertEqual(reason, "inconclusive_evidence")

    def test_ambiguous_content_accepts(self):
        self.assertEqual(decide(0.12, 0.11, ocr=text_evidence())[0], "accept")

    def test_tied_layout_evidence_does_not_reject(self):
        self.assertEqual(decide(0.15, 0.15, ocr=text_evidence("buy now, 50% off", 20))[0], "accept")

    def test_clear_layout_and_promotional_text_reject(self):
        self.assertEqual(decide(0.10, 0.13, ocr=text_evidence("book now, limited offer", 10))[0], "reject")

    def test_text_alone_cannot_reject_a_photographed_scene(self):
        self.assertEqual(decide(0.15, 0.10, ocr=text_evidence("buy now, 50% off", 40))[0], "accept")

    def test_strong_poster_evidence_does_not_require_known_keywords(self):
        self.assertEqual(decide(0.06, 0.12, ocr=text_evidence("unrecognized wording", 12))[0], "reject")

    def test_small_text_or_one_word_is_not_enough(self):
        self.assertEqual(decide(0.10, 0.20, ocr=text_evidence("book now", 1))[0], "accept")
        self.assertEqual(decide(0.10, 0.20, ocr=text_evidence("sale", 25, words=1))[0], "accept")

    def test_low_absolute_cosine_does_not_hide_a_clear_ad(self):
        self.assertEqual(decide(0.03, 0.08, ocr=text_evidence("book now, 4 days 3 nights", 10))[0], "reject")

    def test_close_comparison_remains_conservative(self):
        self.assertEqual(decide(0.08, 0.081, ocr=text_evidence("buy now, 50% off", 25))[0], "accept")

    def test_nonfinite_evidence_is_unknown_not_zero(self):
        for value in (float("nan"), float("inf")):
            decision, score, reason = decide(value, 0.1)
            self.assertEqual((decision, score, reason), ("accept", None, "inconclusive_evidence"))

    def test_no_promotion_keywords_still_produces_a_visual_score(self):
        decision, score, _ = decide(0.12, 0.13, ocr=text_evidence("ordinary caption", 2))
        self.assertEqual(decision, "accept")
        self.assertGreater(score, 50)

    def test_invalid_threshold_is_an_error(self):
        for value in (0, 75, 101, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decide(0.2, 0.1, value)

    def test_higher_threshold_cannot_turn_acceptance_into_rejection(self):
        for ad in (0.11, 0.13, 0.15):
            outcomes = [decide(0.10, ad, cutoff, text_evidence("buy now, 50% off", 10))[0] for cutoff in (90, 95, 99)]
            if "accept" in outcomes:
                self.assertNotIn("reject", outcomes[outcomes.index("accept"):])

    def test_processing_error_is_distinct_from_ad_rejection(self):
        result = failure_result("bad.png", "decode failed")
        self.assertEqual(result["decision"], "error")
        self.assertFalse(result["ad_rejected"])

    def test_invalid_image_fails_decoding(self):
        with self.assertRaises(ValueError):
            load_image(b"not an image")

    def test_exif_orientation_is_applied(self):
        source = Image.new("RGB", (120, 180), "white")
        exif = Image.Exif(); exif[274] = 6
        data = io.BytesIO(); source.save(data, format="JPEG", exif=exif)
        self.assertEqual(load_image(data.getvalue()).size, (180, 120))


class CommercialTextTests(unittest.TestCase):
    def test_restaurant_offers_without_buy_now(self):
        self.assertGreaterEqual(commercial_evidence("Happy hour! $5 meals, $3 drinks, first drink free")['strength'], 0.45)

    def test_travel_package_without_sale_words(self):
        self.assertGreaterEqual(commercial_evidence("4DAYS 3NIGHTS, accommodation, call us on +256 756 666 2222")['strength'], 0.45)

    def test_phone_number_alone_is_weak_evidence(self):
        self.assertLess(commercial_evidence("Contact address: +256 756 666 2222")['strength'], 0.45)

    def test_calendar_and_ordinary_text_are_not_commercial(self):
        self.assertEqual(commercial_evidence("Friday: public park opening times. Trees, water and paths.")['strength'], 0)


class ModelAndHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ContentFilter()
        server.ENGINE = cls.engine
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.http.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown(); cls.http.server_close(); cls.thread.join(timeout=2)

    def image(self, filename):
        return load_image((ROOT / filename).read_bytes())

    def check_image(self, filename, expected):
        result = self.engine.analyze(self.image(filename))
        self.assertEqual(result["decision"], expected, filename)
        self.assertEqual(result["accepted"], expected == "accept")
        self.assertEqual(result["ad_rejected"], expected == "reject")
        self.assertEqual(result["ambiguous_action"], "accept")
        self.assertEqual(result["scope"], "obvious_advertisements_only")
        self.assertFalse(result["waviness_checked"])
        self.assertFalse(result["score_is_probability"])
        json.dumps(result, allow_nan=False)
        return result

    def test_nonadvertisement_regressions_accept(self):
        for filename in ("street.jpg", "street-wavy.png", "street-with-sign.jpg", "landscape.jpg",
                         "park-statue.jpg", "ice-landscape.jpg", "animals.png", "portrait.jpg",
                         "screenshot.jpg", "document.png", "logo.png", "blank.png"):
            with self.subTest(filename=filename):
                self.check_image(filename, "accept")

    def test_obvious_advertising_graphics_reject(self):
        for filename in ("advertisement.png", "advertisement-with-street.png", "advertisement-panorama.png",
                         "advertisement-with-statue.png", "advertisement-with-ice.png",
                         "advertisement-restaurant.png", "advertisement-resort.png"):
            with self.subTest(filename=filename):
                result = self.check_image(filename, "reject")
                self.assertEqual(result["reason_code"], "obvious_advertisement")
                self.assertTrue(result["evidence"]["clear_advertising_layout"])
                self.assertTrue(result["evidence"]["substantial_text"])

    def test_supplied_statue_and_ice_at_reduced_resolution_accept(self):
        for filename in ("park-statue.jpg", "ice-landscape.jpg"):
            with self.subTest(filename=filename):
                image = self.image(filename)
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                self.assertEqual(self.engine.analyze(image)["decision"], "accept")

    def test_missing_ocr_does_not_reject(self):
        with patch("panomoche.advertising.classifier.shutil.which", return_value=None):
            result = self.engine.analyze(self.image("advertisement.png"))
        self.assertEqual(result["decision"], "accept")
        self.assertFalse(result["checked"])
        self.assertEqual(result["reason_code"], "inconclusive_evidence")

    def test_text_is_checked_even_if_visual_match_is_not_advertising(self):
        with patch("panomoche.advertising.classifier.ocr_evidence", return_value={**text_evidence(), "status": "ready", "promotional_terms": []}) as ocr:
            result = self.engine.analyze(self.image("street.jpg"))
        ocr.assert_called_once()
        self.assertTrue(result["accepted"])

    def test_real_ads_have_high_scores_and_explainable_components(self):
        for filename in ("advertisement-restaurant.png", "advertisement-resort.png"):
            with self.subTest(filename=filename):
                result = self.check_image(filename, "reject")
                self.assertGreater(result["ad_score"], 90)
                self.assertGreater(result["score_components"]["commercial_text"], 45)

    def test_photographic_content_without_advertising_overlay_accepts(self):
        for filename, box in (("advertisement-restaurant.png", (0.01, 0.22, 0.61, 0.50)),
                              ("advertisement-resort.png", (0.50, 0.22, 0.95, 0.60))):
            with self.subTest(filename=filename):
                image = self.image(filename)
                crop = image.crop(tuple(round(value * (image.width if i % 2 == 0 else image.height)) for i, value in enumerate(box)))
                self.assertTrue(self.engine.analyze(crop)["accepted"])

    def test_visual_model_failure_passes_unchecked(self):
        with patch.object(self.engine.encoder, "run", side_effect=RuntimeError("test model failure")):
            result = self.engine.analyze(self.image("street.jpg"))
        self.assertTrue(result["accepted"])
        self.assertFalse(result["checked"])
        self.assertEqual(result["detection_status"], "unavailable")

    def post(self, path, payload):
        return urlopen(Request(self.url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}), timeout=30)

    def payload(self, filename="street.jpg", **extra):
        return {"image": base64.b64encode((ROOT / filename).read_bytes()).decode(), "filename": filename, **extra}

    def test_http_accepts_valid_nonad(self):
        with self.post("/api/analyze", self.payload("animals.png")) as response:
            result = json.load(response)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["filename"], "animals.png")

    def test_http_rejects_obvious_ad(self):
        with self.post("/api/analyze", self.payload("advertisement.png")) as response:
            result = json.load(response)
        self.assertTrue(result["ad_rejected"])

    def test_old_place_relevance_threshold_is_ignored(self):
        with self.post("/api/analyze", self.payload(threshold=99)) as response:
            result = json.load(response)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["threshold"], 90)

    def test_ad_threshold_is_applied(self):
        with self.post("/api/analyze", self.payload(ad_threshold=99)) as response:
            result = json.load(response)
        self.assertEqual(result["threshold"], 99)

    def test_bad_upload_is_processing_error_not_ad_rejection(self):
        with self.assertRaises(HTTPError) as caught:
            self.post("/api/analyze", {"image": "invalid!!!!", "filename": "broken.png"})
        result = json.load(caught.exception)
        self.assertEqual(result["decision"], "error")
        self.assertFalse(result["ad_rejected"])
        self.assertEqual(result["filename"], "broken.png")

    def test_invalid_ad_threshold_is_request_error(self):
        with self.assertRaises(HTTPError) as caught:
            self.post("/api/analyze", self.payload(ad_threshold=75))
        self.assertEqual(caught.exception.code, 400)
        self.assertFalse(json.load(caught.exception)["ad_rejected"])

    def test_unavailable_model_passes_valid_input_unchecked(self):
        with patch.object(server, "ENGINE", None):
            with self.post("/api/analyze", self.payload()) as response:
                result = json.load(response)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["checked"])
        self.assertEqual(result["detection_status"], "unavailable")

    def test_unavailable_model_still_validates_input(self):
        with patch.object(server, "ENGINE", None), self.assertRaises(HTTPError) as caught:
            self.post("/api/analyze", {"image": "invalid!!!!"})
        self.assertEqual(caught.exception.code, 400)

    def test_busy_response_is_not_ad_rejection(self):
        server.ANALYSIS_LOCK.acquire()
        try:
            with self.assertRaises(HTTPError) as caught:
                self.post("/api/analyze", self.payload())
            self.assertEqual(caught.exception.code, 429)
            self.assertFalse(json.load(caught.exception)["ad_rejected"])
        finally:
            server.ANALYSIS_LOCK.release()

    def test_json_export_is_a_download(self):
        with self.post("/api/export", {"filename": "report.json", "content": '{"decision":"accept"}'}) as response:
            path = json.load(response)["url"]
        with urlopen(self.url + path, timeout=5) as response:
            self.assertIn("attachment", response.headers["Content-Disposition"])
            self.assertEqual(json.load(response)["decision"], "accept")

    def test_only_ui_assets_are_served(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.url + "/policy.json", timeout=5)
        self.assertEqual(caught.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
