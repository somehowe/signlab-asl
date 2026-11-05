"""Serve SignLab's local demo and its read-only model and evaluation API."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import numpy as np

from model.predict import Predictor

ROOT = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 1_000_000
MAX_FRAMES = 240
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def json_frames(points: np.ndarray) -> list:
    """Replace missing coordinates with JSON nulls."""
    values = points.astype(object)
    values[~np.isfinite(points)] = None
    return values.tolist()


class Demo:
    def __init__(self, directory: Path, web: Path) -> None:
        self.web = web
        self.predictor = Predictor.load(directory / "model.npz")
        self.report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        with np.load(directory / "demo.npz", allow_pickle=False) as archive:
            self.landmarks = archive["landmarks"]
            self.ids = archive["sample_ids"].astype(str).tolist()
            self.labels = archive["labels"].astype(str).tolist()
            self.signers = archive["signers"].astype(str).tolist()
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Demo samples must have unique IDs.")
        self.rows = {sample_id: index for index, sample_id in enumerate(self.ids)}
        self.catalog = {
            "dataset": self.report.get("dataset", "See the data card in the README"),
            "labels": self.report["classes"],
            "point_count": self.landmarks.shape[2],
            "dimensions": self.landmarks.shape[3],
            "samples": [
                {"id": sample_id, "label": label, "signer": signer}
                for sample_id, label, signer in zip(
                    self.ids, self.labels, self.signers, strict=True
                )
            ],
        }

    def predict(self, request: dict) -> dict:
        if not isinstance(request, dict) or set(request) not in (
            {"sample_id"}, {"frames"}
        ):
            raise ValueError("Send either a sample_id or a frames array.")
        if "sample_id" in request:
            sample_id = request["sample_id"]
            if not isinstance(sample_id, str) or sample_id not in self.rows:
                raise LookupError("That sample is not in the demo catalog.")
            row = self.rows[sample_id]
            points = self.landmarks[row]
            truth = self.labels[row]
        else:
            try:
                points = np.asarray(request["frames"], dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Landmarks must be numbers or nulls.") from error
            truth = None

        shape = self.landmarks.shape[2:]
        if points.ndim != 3 or not 1 <= len(points) <= MAX_FRAMES or points.shape[1:] != shape:
            raise ValueError(f"Expected 1 to {MAX_FRAMES} frames of {shape[0]} points with {shape[1]} coordinates.")
        if np.isinf(points).any() or not np.isfinite(points).any():
            raise ValueError("Provide finite coordinates for at least one landmark.")

        started = perf_counter()
        result = self.predictor.predict(points, top_k=min(5, len(self.report["classes"])))
        latency_ms = round((perf_counter() - started) * 1000, 2)
        return {
            **result,
            "prediction": result["label"],
            "scores": [
                {**entry, "score": entry["probability"]} for entry in result["scores"]
            ],
            "frames": json_frames(points),
            "truth": truth,
            "latency_ms": latency_ms,
        }


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, demo: Demo, **kwargs) -> None:
        self.demo = demo
        super().__init__(*args, **kwargs)

    def write_body(self, status: int, body: bytes, media_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def write_json(self, status: int, value: dict) -> None:
        self.write_body(
            status,
            json.dumps(value, allow_nan=False, separators=(",", ":")).encode(),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:
        route = urlsplit(self.path).path
        if route == "/api/catalog":
            self.write_json(200, self.demo.catalog)
        elif route == "/api/report":
            self.write_json(200, self.demo.report)
        elif route in ASSETS:
            name, media_type = ASSETS[route]
            self.write_body(200, (self.demo.web / name).read_bytes(), media_type)
        else:
            self.write_json(404, {"error": "This route does not exist."})

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/predict":
            self.write_json(404, {"error": "This route does not exist."})
            return
        if self.headers.get_content_type() != "application/json":
            self.write_json(415, {"error": "Send application/json."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.write_json(400, {"error": "Invalid request length."})
            return
        if not 1 <= size <= MAX_REQUEST_BYTES:
            self.write_json(413, {"error": "The JSON body is empty or too large."})
            return

        try:
            request = json.loads(self.rfile.read(size))
            result = self.demo.predict(request)
        except LookupError as error:
            self.write_json(404, {"error": str(error)})
        except (ValueError, UnicodeError) as error:
            self.write_json(400, {"error": str(error)})
        else:
            self.write_json(200, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        demo = Demo(args.artifacts, ROOT / "web")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Could not load SignLab artifacts: {error}\n")

    with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, demo=demo)) as server:
        print(f"SignLab is ready at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
