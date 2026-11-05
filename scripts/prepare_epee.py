"""Build a small isolated-gloss benchmark from public CLERC Épée v0.4 keypoints."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from io import BytesIO
import json
from pathlib import Path
import random
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np


REPOSITORY = "https://huggingface.co/datasets/CLERC-DATA/epee"
REVISION = "9037c3f0278a900ac26eb4ba065a44d791d28f98"
BASE_URL = f"{REPOSITORY}/resolve/{REVISION}"
GLOSSES = ("CAN", "HAVE", "NEED", "WANT", "WHAT", "YOU", "YOUR")
SIGNERS = ("ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL")
LANDMARKS = [*range(46), 54, 55]
NUM_FRAMES = 16
SEED = 2026


def fetch(relative_path: str, cache: Path) -> bytes:
    cached = cache / relative_path
    if cached.is_file():
        return cached.read_bytes()
    for attempt in range(3):
        try:
            with urlopen(f"{BASE_URL}/{relative_path}", timeout=18) as response:
                contents = response.read()
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(contents)
            return contents
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def select_clips(cache: Path, per_signer: int) -> list[tuple[dict[str, str], str]]:
    csv_text = fetch("metadata.csv", cache).decode("utf-8")
    rows = list(csv.DictReader(csv_text.splitlines()))
    randomizer = random.Random(SEED)
    selected = []
    for signer in SIGNERS:
        candidates = {
            gloss: [
                row for row in rows
                if row["signer_id"] == signer
                and row["hand_slots"] == "verified"
                and gloss in row["gloss_sequence"].split(" | ")
            ]
            for gloss in GLOSSES
        }
        used_clips = set()
        # Fill rare signer/gloss combinations first, with only one segment per clip.
        for gloss, matches in sorted(candidates.items(), key=lambda item: len(item[1])):
            randomizer.shuffle(matches)
            unused = [row for row in matches if row["clip_id"] not in used_clips]
            for row in unused[:per_signer]:
                used_clips.add(row["clip_id"])
                selected.append((row, gloss))
    return sorted(selected, key=lambda item: item[0]["clip_id"])


def load_segment(row: dict[str, str], gloss: str, cache: Path) -> tuple:
    clip_id = row["clip_id"]
    annotation = json.loads(fetch(f"annotations/{clip_id}.json", cache))
    raw = fetch(f"keypoints/{clip_id}.npy", cache)
    points = np.load(BytesIO(raw), allow_pickle=False)
    if points.shape != (int(row["n_frames"]), 128, 3):
        raise ValueError(f"Unexpected keypoint shape for {clip_id}: {points.shape}")
    if annotation["signer_id"] != row["signer_id"]:
        raise ValueError(f"Signer metadata mismatch for {clip_id}")
    matches = [segment for segment in annotation["segments"] if segment["gloss"] == gloss]
    segment = max(matches, key=lambda match: match["end"] - match["start"])
    fps = float(annotation["fps"])
    start = max(0, round(segment["start"] * fps))
    stop = min(len(points), round(segment["end"] * fps))
    if stop - start < 2:
        raise ValueError(f"Fewer than two annotated frames in {clip_id}")
    indices = np.rint(np.linspace(start, stop - 1, NUM_FRAMES)).astype(int)
    clip = points[indices][:, LANDMARKS].astype(np.float32, copy=True)
    clip[np.all(clip == 0, axis=-1)] = np.nan
    return (
        clip_id, gloss, row["signer_id"], clip, float(segment["start"]),
        float(segment["end"]), int(row["phrase_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("epee_glosses.npz"))
    parser.add_argument("--cache", type=Path, default=Path(".epee-cache"))
    parser.add_argument("--per-signer", type=int, default=6)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.per_signer < 2 or args.workers < 1:
        parser.error("--per-signer must be at least two and --workers must be positive")
    selected = select_clips(args.cache, args.per_signer)
    print(f"Downloading annotations and keypoints for {len(selected)} distinct clips", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [
            pool.submit(load_segment, row, gloss, args.cache)
            for row, gloss in selected
        ]
        loaded = []
        for index, future in enumerate(as_completed(pending), start=1):
            loaded.append(future.result())
            if index % 40 == 0:
                print(f"Prepared {index}/{len(selected)} clips", flush=True)
    loaded.sort(key=lambda item: item[0])
    clip_ids, labels, signers, points, starts, ends, phrases = zip(*loaded)
    assert len(set(clip_ids)) == len(clip_ids)
    counts = Counter(zip(signers, labels))
    if set(counts) != {(s, g) for s in SIGNERS for g in GLOSSES}:
        raise ValueError("The resulting subset must cover every signer and gloss combination")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        landmarks=np.stack(points),
        labels=np.asarray(labels),
        signers=np.asarray(signers),
        sample_ids=np.asarray(clip_ids),
        source_clip_ids=np.asarray(clip_ids),
        source_start_seconds=np.asarray(starts),
        source_end_seconds=np.asarray(ends),
        phrase_ids=np.asarray(phrases),
        hand_indices=np.arange(42),
        pose_indices=np.arange(42, len(LANDMARKS)),
        shoulder_indices=np.array([42, 43]),
        source_landmark_indices=np.asarray(LANDMARKS),
    )
    print(f"Wrote {args.out} with {len(points)} samples, {len(set(signers))} signers and {len(set(labels))} glosses")
    for gloss in GLOSSES:
        print(gloss, *(counts[(signer, gloss)] for signer in SIGNERS))


if __name__ == "__main__":
    main()
