from __future__ import annotations

import unittest

import numpy as np

from mention_graph.diagnostic_pipeline import (
    _crossed_bootstrap,
    _evidence_inputs,
    _repeat_masks,
)
from mention_graph.relational import build_relational_source
from mention_graph.schema import Document, Mention


def _document() -> Document:
    mentions = [
        Mention("m1", "d1", 0, 4, "Київ", "LOC", 0, 0, 20),
        Mention("m2", "d1", 6, 10, "Київ", "LOC", 0, 0, 20),
        Mention("m3", "d1", 12, 16, "Львів", "LOC", 0, 0, 20),
    ]
    return Document(
        doc_id="d1",
        source="test",
        text="Київ, Київ, Львів.",
        sentence_spans=[(0, 20)],
        mentions=mentions,
        split="test",
    )


class EvidenceAndTargetTests(unittest.TestCase):
    def test_repeat_target_selection_is_one_per_group_and_label_free(self) -> None:
        mentions = _document().mentions
        target, component, counts = _repeat_masks(mentions, target_seed=7)
        self.assertEqual(int(target.sum()), 1)
        self.assertEqual(int(component.sum()), 2)
        self.assertEqual(counts["groups"], 1)
        mentions[0].label = "ORG"
        changed_target, changed_component, _ = _repeat_masks(
            mentions,
            target_seed=7,
        )
        np.testing.assert_array_equal(target, changed_target)
        np.testing.assert_array_equal(component, changed_component)

    def test_probability_feature_slice_matches_residual_channel(self) -> None:
        document = _document()
        source = build_relational_source(
            [document],
            label_to_id={"LOC": 0},
            near_threshold_tokens=20,
            alias_threshold=0.9,
        )
        clean = np.arange(12, dtype=np.float32).reshape(3, 4)
        surface = clean + 100
        probabilities = np.ones((3, 1), dtype=np.float32)
        inputs = _evidence_inputs(
            clean,
            surface,
            probabilities,
            probabilities,
            source,
            mode="surface",
            evidence_mask=np.asarray([True, False, False]),
            evaluation_mask=np.asarray([True, False, False]),
            train_prior=np.asarray([1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            inputs.features[:, 4:5],
            inputs.local_probabilities,
        )


class CrossedBootstrapTests(unittest.TestCase):
    def test_interaction_uses_paired_realizations_and_documents(self) -> None:
        truth = np.asarray([0, 1, 0, 1], dtype=np.int64)
        store: dict[tuple[int, str, int, str], np.ndarray] = {}
        masks: dict[tuple[str, int], np.ndarray] = {}
        for seed in (1, 2):
            for realization in (10, 20):
                masks[("weak", realization)] = np.ones(4, dtype=bool)
                masks[("clean", realization)] = np.ones(4, dtype=bool)
                store[(seed, "weak", realization, "N0")] = np.asarray(
                    [1, 0, 1, 0]
                )
                store[(seed, "weak", realization, "G")] = truth.copy()
                store[(seed, "clean", realization, "N0")] = truth.copy()
                store[(seed, "clean", realization, "G")] = truth.copy()
        rows = _crossed_bootstrap(
            truth,
            store,
            masks,
            ["d1", "d1", "d2", "d2"],
            {"d1": "a", "d2": "b"},
            ["A", "B"],
            [
                {
                    "family": "test",
                    "contrast": "interaction",
                    "metric": "accuracy",
                    "sesoi": 0.01,
                    "terms": [
                        {
                            "coefficient": 1,
                            "condition": "weak",
                            "model": "G",
                        },
                        {
                            "coefficient": -1,
                            "condition": "weak",
                            "model": "N0",
                        },
                        {
                            "coefficient": -1,
                            "condition": "clean",
                            "model": "G",
                        },
                        {
                            "coefficient": 1,
                            "condition": "clean",
                            "model": "N0",
                        },
                    ],
                }
            ],
            samples=100,
            seed=3,
        )
        self.assertAlmostEqual(rows[0]["observed_mean_contrast"], 1.0)
        self.assertEqual(rows[0]["realizations"], 2)


if __name__ == "__main__":
    unittest.main()
