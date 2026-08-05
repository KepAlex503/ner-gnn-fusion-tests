from __future__ import annotations

import unittest

import numpy as np

from mention_graph.metrics import (
    classification_metrics,
    paired_document_bootstrap,
    probability_metrics,
)
from mention_graph.training import document_folds


class MetricTests(unittest.TestCase):
    def test_micro_f1_equals_accuracy_for_single_label_multiclass(self) -> None:
        truth = np.asarray([0, 1, 2, 2, 1])
        prediction = np.asarray([0, 2, 2, 2, 1])
        metrics = classification_metrics(
            truth,
            prediction,
            ["A", "B", "C"],
        )
        self.assertAlmostEqual(metrics["accuracy"], metrics["micro_f1"])

    def test_document_bootstrap_keeps_documents_as_units(self) -> None:
        truth = np.asarray([0, 1, 0, 1])
        baseline = np.asarray([0, 0, 1, 1])
        graph = np.asarray([0, 1, 0, 1])
        result = paired_document_bootstrap(
            truth,
            baseline,
            graph,
            ["d1", "d1", "d2", "d2"],
            ["A", "B"],
            samples=200,
            seed=5,
        )
        self.assertEqual(result["documents"], 2)
        self.assertGreater(result["observed_delta_macro_f1"], 0)

    def test_perfect_probabilities_are_calibrated(self) -> None:
        truth = np.asarray([0, 1, 0])
        probabilities = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            dtype=np.float64,
        )
        metrics = probability_metrics(truth, probabilities)
        self.assertAlmostEqual(metrics["brier_score"], 0.0)
        self.assertAlmostEqual(metrics["ece_10_bins"], 0.0)


class FoldTests(unittest.TestCase):
    def test_each_document_belongs_to_exactly_one_oof_fold(self) -> None:
        document_ids = ["a", "a", "b", "b", "c", "d", "e", "f"]
        sources = {
            "a": "x",
            "b": "x",
            "c": "x",
            "d": "y",
            "e": "y",
            "f": "y",
        }
        assignments = document_folds(
            document_ids,
            sources,
            fold_count=3,
            seed=9,
        )
        self.assertEqual(set(assignments), set(document_ids))
        self.assertTrue(all(0 <= fold < 3 for fold in assignments.values()))
        self.assertEqual(assignments["a"], assignments["a"])


if __name__ == "__main__":
    unittest.main()
