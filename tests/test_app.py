"""Smoke tests against the released model, report, and held-out demo set."""

from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import Demo, Handler

ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifacts = ROOT / "artifacts"
        if not (artifacts / "demo.npz").exists():
            raise unittest.SkipTest("Run scripts/prepare_demo.py before testing the app.")
        demo = Demo(artifacts, ROOT / "web")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, demo=demo))
        cls.worker = Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=5)

    def fetch(self, route, body=None):
        request = Request(
            self.base + route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def test_catalog_uses_held_out_clips_and_predictions_round_trip(self):
        report = self.fetch("/api/report")
        catalog = self.fetch("/api/catalog")
        self.assertTrue(catalog["samples"])
        self.assertEqual(report["protocol"], "signer-disjoint")
        self.assertTrue(
            {sample["id"] for sample in catalog["samples"]} <= set(report["held_out_ids"])
        )
        sample = catalog["samples"][0]
        result = self.fetch("/api/predict", {"sample_id": sample["id"]})
        self.assertEqual(result["truth"], sample["label"])
        repeated = self.fetch("/api/predict", {"frames": result["frames"]})
        self.assertEqual(result["scores"], repeated["scores"])
        self.assertEqual(result["prediction"], repeated["prediction"])

    def test_invalid_input_and_missing_sample(self):
        for payload, status in (
            ({"frames": [[[1, 2, 3]]]}, 400),
            ({"sample_id": "missing"}, 404),
        ):
            with self.subTest(payload=payload), self.assertRaises(HTTPError) as caught:
                self.fetch("/api/predict", payload)
            self.assertEqual(caught.exception.code, status)


if __name__ == "__main__":
    unittest.main()
