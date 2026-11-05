import tempfile
import unittest
from pathlib import Path

import numpy as np

from .features import Layout, feature_pair
from .predict import Predictor
from .splits import split_indices
from .train import train


class ModelTests(unittest.TestCase):
    def test_group_split_never_reuses_a_signer(self):
        signers = np.repeat(np.arange(8), 6)
        labels = np.tile(["CAN", "NEED", "YOU"], 16)
        splits = split_indices(labels, signers, 42)
        seen = set()
        for split in splits:
            group_ids = set(signers[split])
            self.assertFalse(seen & group_ids)
            self.assertEqual(set(labels[split]), {"CAN", "NEED", "YOU"})
            seen.update(group_ids)
        self.assertEqual(seen, set(signers))

    def test_features_are_invariant_to_uniform_scale_and_translation(self):
        clip = np.random.default_rng(7).normal(size=(5, 8, 3))
        layout = Layout(8, 3, shoulder_indices=(0, 1))
        expected = feature_pair(clip, layout)
        actual = feature_pair(clip * 3.5 + [2.5, -1.2, 3.0], layout)
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(left, right, atol=1e-12)

    def test_numerical_model_round_trip_and_no_signal_rejection(self):
        random = np.random.default_rng(2026)
        signers = np.repeat(np.arange(6), 6)
        labels = np.tile(["HAVE", "WANT"], 18)
        frames = random.normal(scale=0.025, size=(len(labels), 5, 6, 2))
        frames += np.array([[0, 0], [1, 0], [.3, .6], [.2, .5], [.5, .7], [.7, .9]])
        frames[labels == "WANT", :, 2:4, 1] *= -1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            np.savez(path / "fixture.npz", landmarks=frames, labels=labels, signers=signers)
            report = train(path / "fixture.npz", path, 42)
            predictor = Predictor.load(path / "model.npz")
            self.assertEqual(report["protocol"], "signer-disjoint")
            actual = [
                predictor.predict(frames[i], top_k=2) for i in report["held_out_indices"]
            ]
            predicted = [result["scores"][0]["label"] for result in actual]
            expected = labels[report["held_out_indices"]]
            self.assertAlmostEqual(np.mean(predicted == expected), report["test"]["accuracy"])
            self.assertAlmostEqual(np.mean([not result["abstained"] for result in actual]),
                                   report["test"]["coverage"])
            no_signal = predictor.predict(np.full((3, 6, 2), np.nan))
            self.assertTrue(no_signal["abstained"])
            self.assertEqual(no_signal["scores"], [])


if __name__ == "__main__":
    unittest.main()
