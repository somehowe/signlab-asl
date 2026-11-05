"""Train and evaluate two isolated-sign classifiers on a labeled landmark corpus."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .features import Layout, LowSignal, feature_pair
from .splits import SPLIT_NAMES, split_indices


TARGET_ACCURACY = 0.90
MIN_COVERAGE = 0.60


def text_vector(data, key, count, fallback=None):
    if key not in data:
        return fallback
    values = data[key].astype(str)
    if values.shape != (count,):
        raise ValueError(f"{key} must contain one string per clip.")
    return np.char.strip(values)


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        frames = data["landmarks"]
        if frames.ndim != 4 or len(frames) < 3:
            raise ValueError("Expected landmarks with shape [clip, time, point, coordinate].")
        count, _, points, dimensions = frames.shape
        labels = text_vector(data, "labels", count)
        if labels is None or np.any(labels == "") or len(set(labels)) < 2:
            raise ValueError("Expected nonempty labels for at least two distinct signs.")
        ids = text_vector(data, "sample_ids", count, np.arange(count).astype(str))
        if np.any(ids == "") or len(set(ids)) != count:
            raise ValueError("Expected a unique, nonempty sample ID for every clip.")
        signers = text_vector(data, "signers", count)
        source_clips = text_vector(data, "source_clip_ids", count)
        for key, groups in (("signers", signers), ("source_clip_ids", source_clips)):
            if groups is not None and np.any(groups == "") and np.any(groups != ""):
                raise ValueError(f"Provide {key} for every clip or omit the field.")
        signers = signers if signers is not None and np.any(signers != "") else None
        source_clips = (
            source_clips if source_clips is not None and np.any(source_clips != "") else None
        )
        if signers is not None and source_clips is not None:
            clip_signers = {}
            for clip, signer in zip(source_clips, signers):
                if clip_signers.setdefault(clip, signer) != signer:
                    raise ValueError("A source recording appears under multiple signer IDs.")

        hands = data["hand_indices"].reshape(-1, 21) if "hand_indices" in data else []
        shoulders = data["shoulder_indices"] if "shoulder_indices" in data else []
        layout = Layout(
            points, dimensions, tuple(tuple(map(int, hand)) for hand in hands),
            tuple(map(int, shoulders)),
        )
    return frames, labels, ids, signers, source_clips, layout


def evaluate(model, x, y, threshold):
    probabilities = model.predict_proba(x)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = predictions == y
    accepted = confidence >= threshold
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=model.classes_, zero_division=0
    )
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1.mean()),
        "coverage": float(accepted.mean()),
        "selective_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "accepted": int(accepted.sum()),
        "clips": len(y),
        "confusion": confusion_matrix(y, predictions, labels=model.classes_).tolist(),
        "per_class": [
            {"label": str(label), "precision": float(p), "recall": float(r),
             "f1": float(f), "support": int(n)}
            for label, p, r, f, n in zip(model.classes_, precision, recall, f1, support)
        ],
    }


def validation_threshold(model, x, y):
    """Select the widest coverage reaching the target on validation clips."""
    probabilities = model.predict_proba(x)
    scores = probabilities.max(axis=1)
    correct = model.classes_[probabilities.argmax(axis=1)] == y
    for cutoff in np.unique(np.r_[0.0, scores]):
        accepted = scores >= cutoff
        if accepted.mean() >= MIN_COVERAGE and correct[accepted].mean() >= TARGET_ACCURACY:
            return float(cutoff), True

    feasible = []
    for cutoff in np.unique(np.r_[0.0, scores]):
        accepted = scores >= cutoff
        if accepted.mean() >= MIN_COVERAGE:
            feasible.append((float(correct[accepted].mean()), float(accepted.mean()), -float(cutoff)))
    return -max(feasible)[2], False


def fit_classifier(x, y):
    return LogisticRegression(C=0.25, max_iter=1000).fit(x, y)


def fit(features, labels, train, validation):
    scaler = StandardScaler().fit(features[train])
    x = scaler.transform(features)
    model = fit_classifier(x[train], labels[train])
    validation_open = evaluate(model, x[validation], labels[validation], 0.0)
    return scaler, x, model, validation_open


def signer_audit(feature_sets, labels, signers, data_digest):
    """Compare both fixed models on every signer without choosing a cutoff."""
    results = {}
    with threadpool_limits(limits=2):
        for mode, features in feature_sets.items():
            predictions = np.empty_like(labels)
            folds = []
            for signer in np.unique(signers):
                test = signers == signer
                scaler = StandardScaler().fit(features[~test])
                model = fit_classifier(scaler.transform(features[~test]), labels[~test])
                predictions[test] = model.predict(scaler.transform(features[test]))
                expected = labels[test]
                present = np.unique(expected)
                folds.append({
                    "signer": str(signer),
                    "clips": int(test.sum()),
                    "accuracy": float((predictions[test] == expected).mean()),
                    "macro_f1": float(f1_score(
                        expected, predictions[test], labels=present, average="macro",
                        zero_division=0,
                    )),
                    "labels_present": present.tolist(),
                })
            results[mode] = {
                "pooled_accuracy": float((predictions == labels).mean()),
                "pooled_macro_f1": float(
                    f1_score(labels, predictions, average="macro", zero_division=0)
                ),
                "mean_signer_macro_f1": float(np.mean([fold["macro_f1"] for fold in folds])),
                "folds": folds,
            }
    return {
        "protocol": "leave-one-signer-out",
        "data_sha256": data_digest,
        "scope": (
            "Fixed classifier settings with no confidence cutoff. Each fold's macro F1 "
            "averages signs present for that signer. Pooled macro F1 uses all out-of-fold "
            "predictions. These scores are separate from the main test split."
        ),
        "models": results,
    }


def train(data_path, output_path, seed, *, audit_signers=False):
    frames, labels, ids, signers, source_clips, layout = load_dataset(data_path)
    static, temporal, valid = [], [], []
    for index, clip in enumerate(frames):
        try:
            pooled, ordered = feature_pair(clip, layout)
        except LowSignal:
            continue
        static.append(pooled)
        temporal.append(ordered)
        valid.append(index)

    valid = np.asarray(valid, dtype=int)
    excluded = len(labels) - len(valid)
    if len(set(labels[valid])) != len(set(labels)):
        raise ValueError("At least one sign has no usable clips after visibility checks.")
    labels = labels[valid]
    ids = ids[valid]
    if signers is not None:
        protocol, groups = "signer-disjoint", signers[valid]
    elif source_clips is not None:
        protocol, groups = "source-clip-disjoint", source_clips[valid]
    else:
        protocol, groups = "clip-stratified", None
    train_idx, val_idx, test_idx = split_indices(labels, groups, seed)

    feature_sets = {"static": np.asarray(static), "temporal": np.asarray(temporal)}
    fitted = {}
    with threadpool_limits(limits=2):
        for mode, features in feature_sets.items():
            fitted[mode] = fit(features, labels, train_idx, val_idx)
        chosen = max(fitted, key=lambda mode: fitted[mode][3]["macro_f1"])
        scaler, x, model, _ = fitted[chosen]
        threshold, hit_target = validation_threshold(model, x[val_idx], labels[val_idx])
        validation = evaluate(model, x[val_idx], labels[val_idx], threshold)
        results = {}
        for mode, (_, inputs, candidate, valid_metrics) in fitted.items():
            results[mode] = {
                "name": "Mean pose" if mode == "static" else "Temporal landmarks",
                "validation": valid_metrics,
                "test": evaluate(
                    candidate, inputs[test_idx], labels[test_idx],
                    threshold if mode == chosen else 0.0,
                ),
            }
        test = results[chosen]["test"]

    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path / "model.npz", schema_version=1, mode=chosen, points=layout.points,
        dimensions=layout.dimensions, steps=layout.steps,
        hand_indices=np.asarray(layout.hand_indices, dtype=int).reshape(-1, 21),
        shoulder_indices=np.asarray(layout.shoulder_indices, dtype=int),
        classes=model.classes_, mean=scaler.mean_, scale=scaler.scale_,
        weights=model.coef_, intercept=model.intercept_, threshold=threshold,
    )
    digest = sha256()
    with data_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            digest.update(chunk)
    data_digest = digest.hexdigest()
    if audit_signers:
        if signers is None:
            raise ValueError("A signer audit requires signer IDs.")
        audit = signer_audit(feature_sets, labels, signers[valid], data_digest)
        (output_path / "signer_audit.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    report = {
        "schema_version": 1,
        "data_sha256": data_digest,
        "protocol": protocol,
        "seed": seed,
        "split_counts": {
            name: {
                "clips": len(indices),
                "signers": len(np.unique(signers[valid[indices]])) if signers is not None else None,
                "source_clips": (
                    len(np.unique(source_clips[valid[indices]])) if source_clips is not None else None
                ),
            }
            for name, indices in zip(SPLIT_NAMES, (train_idx, val_idx, test_idx))
        },
        "split_signers": {
            name: np.unique(signers[valid[indices]]).tolist()
            for name, indices in zip(SPLIT_NAMES, (train_idx, val_idx, test_idx))
        } if signers is not None else None,
        "classes": model.classes_.tolist(),
        "comparison": [
            {"name": candidate["name"], "split": "test", "macro_f1": candidate["test"]["macro_f1"],
             "accuracy": candidate["test"]["accuracy"], "n": candidate["test"]["clips"]}
            for candidate in results.values()
        ],
        "models": results,
        "selected_model": chosen,
        "threshold": threshold,
        "validation_target": TARGET_ACCURACY,
        "validation_min_coverage": MIN_COVERAGE,
        "validation_target_met": hit_target,
        "validation": validation,
        "test": test,
        "confusion": test["confusion"],
        "held_out_indices": valid[test_idx].tolist(),
        "held_out_ids": ids[test_idx].tolist(),
        "excluded_low_signal": excluded,
    }
    (output_path / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audit-signer-cv", action="store_true",
        help="Also write a leave-one-signer-out comparison to signer_audit.json.",
    )
    args = parser.parse_args()
    try:
        report = train(args.data, args.output, args.seed, audit_signers=args.audit_signer_cv)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    metrics = report["test"]
    print(
        f"{report['protocol']} test: {metrics['clips']} clips, "
        f"macro F1 {metrics['macro_f1']:.3f}, accuracy {metrics['accuracy']:.3f}, "
        f"coverage {metrics['coverage']:.3f}"
    )


if __name__ == "__main__":
    main()
