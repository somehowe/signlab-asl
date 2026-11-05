"""Select fixed examples from the held-out test set for the SignLab app."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

DATASET = "CLERC Épée v0.4 · timed ASL gloss segments"
DATASET_URL = "https://huggingface.co/datasets/CLERC-DATA/epee"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--examples-per-gloss-signer", type=int, default=2)
    args = parser.parse_args()
    if args.examples_per_gloss_signer < 1:
        parser.error("--examples-per-gloss-signer must be positive")

    report_path = args.artifacts / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    test_ids = set(report["held_out_ids"])
    counts: Counter[tuple[str, str]] = Counter()
    with np.load(args.data, allow_pickle=False) as data:
        labels = data["labels"].astype(str)
        signers = data["signers"].astype(str)
        sample_ids = data["sample_ids"].astype(str)
        if not test_ids or not test_ids.issubset(sample_ids):
            raise ValueError("Evaluation and sample data have different held-out IDs.")

        selected = []
        order = sorted(
            np.flatnonzero(np.isin(sample_ids, list(test_ids))),
            key=lambda i: (labels[i], signers[i], sample_ids[i]),
        )
        for index in order:
            key = (labels[index], signers[index])
            if counts[key] < args.examples_per_gloss_signer:
                counts[key] += 1
                selected.append(index)

        output = args.artifacts / "demo.npz"
        fields = ("landmarks", "labels", "signers", "sample_ids")
        np.savez_compressed(output, **{field: data[field][selected] for field in fields})

    report.update(
        dataset=DATASET,
        dataset_url=DATASET_URL,
        demo_selection="Up to two clips per gloss and test signer, ordered by source clip ID.",
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} held-out examples to {output}.")


if __name__ == "__main__":
    main()
