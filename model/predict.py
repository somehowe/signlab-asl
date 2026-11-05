"""Load the numerical model artifact and score a landmark sequence with NumPy."""

from pathlib import Path

import numpy as np

from .features import Layout, LowSignal, feature_pair


class Predictor:
    def __init__(self, layout, classes, mean, scale, weights, intercept, threshold, mode):
        self.layout = layout
        self.classes = classes
        self.mean = mean
        self.scale = scale
        self.weights = weights
        self.intercept = intercept
        self.threshold = threshold
        self.mode = mode

    @classmethod
    def load(cls, path: str | Path):
        """Load an NPZ produced by model.train without deserializing executable objects."""
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["schema_version"]) != 1:
                raise ValueError("Unsupported model artifact version.")
            layout = Layout(
                points=int(saved["points"]),
                dimensions=int(saved["dimensions"]),
                hand_indices=tuple(tuple(map(int, hand)) for hand in saved["hand_indices"]),
                shoulder_indices=tuple(map(int, saved["shoulder_indices"])),
                steps=int(saved["steps"]),
            )
            return cls(
                layout, saved["classes"].astype(str).tolist(), saved["mean"],
                saved["scale"], saved["weights"], saved["intercept"],
                float(saved["threshold"]), str(saved["mode"]),
            )

    def predict(self, frames, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        try:
            static, temporal = feature_pair(frames, self.layout)
        except LowSignal as error:
            return {
                "label": None, "confidence": 0.0, "threshold": self.threshold,
                "abstained": True, "scores": [], "reason": str(error),
            }
        feature = static if self.mode == "static" else temporal
        logits = ((feature - self.mean) / self.scale) @ self.weights.T + self.intercept
        if len(self.classes) == 2:
            logits = np.array([0.0, float(logits[0])])
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        order = np.argsort(-probabilities, kind="stable")
        best = int(order[0])
        confidence = float(probabilities[best])
        accepted = confidence >= self.threshold
        return {
            "label": self.classes[best] if accepted else None,
            "confidence": confidence,
            "threshold": self.threshold,
            "abstained": not accepted,
            "reason": None if accepted else "The top score fell below the validation cutoff.",
            "scores": [
                {"label": self.classes[index], "probability": float(probabilities[index])}
                for index in order[:top_k]
            ],
        }
