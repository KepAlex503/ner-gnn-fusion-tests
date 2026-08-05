from __future__ import annotations

import unittest

import numpy as np

from mention_graph.diagnostic_controls import (
    build_evidence_view,
    build_sparse_gold_oracle_edge_indices,
    build_train_surface_label_stats,
    degree_preserving_randomize_edge_indices_by_document,
    repeat_ambiguity_cohort_masks,
    semantic_union_pruned_to_same_label_edge_indices,
    stable_nested_document_subsets,
    stable_nested_mention_mask,
    stable_nested_mention_masks,
)
from mention_graph.schema import Document, Mention


def _mention(
    mention_id: str,
    doc_id: str,
    start: int,
    text: str,
    label: str,
) -> Mention:
    return Mention(
        mention_id=mention_id,
        doc_id=doc_id,
        start=start,
        end=start + len(text),
        text=text,
        label=label,
        sentence_id=0,
        sentence_start=0,
        sentence_end=1000,
    )


def _document(
    doc_id: str,
    source: str,
    split: str,
    mentions: list[Mention] | None = None,
) -> Document:
    return Document(
        doc_id=doc_id,
        source=source,
        text="",
        sentence_spans=[(0, 1000)],
        mentions=[] if mentions is None else mentions,
        split=split,
    )


def _undirected(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(left), int(right))
        for left, right in edge_index.T
        if int(left) < int(right)
    }


def _degrees(edge_index: np.ndarray, node_count: int) -> np.ndarray:
    return np.bincount(edge_index[1], minlength=node_count)


class OracleGraphTests(unittest.TestCase):
    def test_sparse_oracle_is_same_label_intra_document_and_one_channel(self) -> None:
        labels = np.asarray(["A", "A", "B", "A", "A", "A", "A", "B"])
        document_ids = ["d1", "d1", "d1", "d1", "d1", "d2", "d2", "d2"]
        positions = [40, 0, 10, 20, 30, 5, 15, 25]
        edge_indices = build_sparse_gold_oracle_edge_indices(
            labels,
            document_ids,
            positions=positions,
        )

        self.assertGreater(edge_indices["sent"].shape[1], 0)
        for relation in ("repeat", "near", "alias"):
            self.assertEqual(edge_indices[relation].shape, (2, 0))
        for left, right in _undirected(edge_indices["sent"]):
            self.assertEqual(document_ids[left], document_ids[right])
            self.assertEqual(labels[left], labels[right])
        self.assertLessEqual(int(_degrees(edge_indices["sent"], len(labels)).max()), 4)
        self.assertEqual(
            _undirected(edge_indices["sent"]),
            {(0, 3), (0, 4), (1, 3), (1, 4), (3, 4), (5, 6)},
        )
        directed_edges = {
            (int(left), int(right))
            for left, right in edge_indices["sent"].T
        }
        self.assertEqual(len(directed_edges), edge_indices["sent"].shape[1])
        self.assertTrue(
            all((right, left) in directed_edges for left, right in directed_edges)
        )
        self.assertTrue(all(left != right for left, right in directed_edges))
        self.assertEqual(edge_indices["sent"].dtype, np.int64)

    def test_per_document_randomization_preserves_every_degree(self) -> None:
        labels = np.asarray(["A"] * 10 + ["B"] * 9)
        document_ids = ["d1"] * 10 + ["d2"] * 9
        oracle = build_sparse_gold_oracle_edge_indices(labels, document_ids)
        randomized = degree_preserving_randomize_edge_indices_by_document(
            oracle,
            document_ids,
            seed=17,
        )
        repeated = degree_preserving_randomize_edge_indices_by_document(
            oracle,
            document_ids,
            seed=17,
        )

        np.testing.assert_array_equal(
            _degrees(oracle["sent"], len(labels)),
            _degrees(randomized["sent"], len(labels)),
        )
        np.testing.assert_array_equal(randomized["sent"], repeated["sent"])
        self.assertEqual(
            oracle["sent"].shape[1],
            randomized["sent"].shape[1],
        )
        self.assertNotEqual(
            _undirected(oracle["sent"]),
            _undirected(randomized["sent"]),
        )
        for left, right in _undirected(randomized["sent"]):
            self.assertEqual(document_ids[left], document_ids[right])
        for relation in ("repeat", "near", "alias"):
            self.assertEqual(randomized[relation].shape, (2, 0))

    def test_semantic_union_is_pruned_to_gold_agreement(self) -> None:
        labels = np.asarray(["A", "A", "B", "B", "A", "B", "A"])
        document_ids = ["d1", "d1", "d1", "d1", "d2", "d2", "d1"]

        def directed(*pairs: tuple[int, int]) -> np.ndarray:
            values = [
                directed_pair
                for left, right in pairs
                for directed_pair in ((left, right), (right, left))
            ]
            return np.asarray(values, dtype=np.int64).T

        semantic = {
            "sent": directed((0, 1), (1, 2), (2, 3)),
            "repeat": directed((0, 2), (4, 5)),
            "near": directed((1, 3)),
            "alias": directed((0, 3)),
        }
        pruned = semantic_union_pruned_to_same_label_edge_indices(
            semantic,
            labels,
            document_ids,
        )
        self.assertEqual(_undirected(pruned["sent"]), {(0, 1), (2, 3)})
        self.assertNotIn((0, 6), _undirected(pruned["sent"]))
        self.assertNotIn((1, 6), _undirected(pruned["sent"]))
        for relation in ("repeat", "near", "alias"):
            self.assertEqual(pruned[relation].shape, (2, 0))


class StableSelectionTests(unittest.TestCase):
    def test_mention_masks_are_nested_and_row_order_independent(self) -> None:
        mention_ids = [f"m-{index:03d}" for index in range(200)]
        masks = stable_nested_mention_masks(
            mention_ids,
            [0.75, 0.25, 0.50],
            seed=29,
        )
        self.assertTrue(np.all(~masks[0.25] | masks[0.50]))
        self.assertTrue(np.all(~masks[0.50] | masks[0.75]))
        self.assertGreater(int(masks[0.50].sum()), 0)
        self.assertLess(int(masks[0.50].sum()), len(mention_ids))
        np.testing.assert_array_equal(
            stable_nested_mention_mask(mention_ids, 0.0, seed=29),
            np.zeros(len(mention_ids), dtype=bool),
        )
        np.testing.assert_array_equal(
            stable_nested_mention_mask(mention_ids, 1.0, seed=29),
            np.ones(len(mention_ids), dtype=bool),
        )

        reversed_ids = list(reversed(mention_ids))
        reversed_mask = stable_nested_mention_mask(
            reversed_ids,
            0.50,
            seed=29,
        )
        decisions = dict(zip(mention_ids, masks[0.50], strict=True))
        reversed_decisions = dict(
            zip(reversed_ids, reversed_mask, strict=True)
        )
        self.assertEqual(decisions, reversed_decisions)

    def test_document_subsets_are_nested_source_stratified_and_train_only(
        self,
    ) -> None:
        documents = [
            *[
                _document(f"a-{index}", "a", "train")
                for index in range(6)
            ],
            *[
                _document(f"b-{index}", "b", "train")
                for index in range(4)
            ],
            _document("validation-leak", "a", "validation"),
            _document("test-leak", "b", "test"),
        ]
        subsets = stable_nested_document_subsets(
            documents,
            [1.0, 0.25, 0.50],
            seed=43,
        )
        quarter = set(subsets[0.25])
        half = set(subsets[0.50])
        full = set(subsets[1.0])
        self.assertLessEqual(quarter, half)
        self.assertLessEqual(half, full)
        self.assertEqual(len(quarter), 3)
        self.assertEqual(len(half), 5)
        self.assertEqual(len(full), 10)
        self.assertNotIn("validation-leak", full)
        self.assertNotIn("test-leak", full)
        self.assertTrue(any(document_id.startswith("a-") for document_id in quarter))
        self.assertTrue(any(document_id.startswith("b-") for document_id in quarter))

        reordered = stable_nested_document_subsets(
            list(reversed(documents)),
            [0.25, 0.50, 1.0],
            seed=43,
        )
        self.assertEqual(subsets, reordered)


class EvidenceViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clean_embeddings = np.asarray(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=np.float32,
        )
        self.surface_embeddings = self.clean_embeddings + 100.0
        self.clean_probabilities = np.asarray(
            [[2.0, 1.0], [1.0, 3.0], [4.0, 1.0], [3.0, 2.0]],
            dtype=np.float32,
        )
        self.surface_probabilities = np.asarray(
            [[1.0, 4.0], [9.0, 1.0], [2.0, 3.0], [1.0, 8.0]],
            dtype=np.float32,
        )
        self.mask = np.asarray([False, True, False, True])
        self.ids = ["m0", "m1", "m2", "m3"]

    def test_surface_view_modifies_both_aligned_evidence_channels(self) -> None:
        view = build_evidence_view(
            self.clean_embeddings,
            self.clean_probabilities,
            mode="surface_only",
            mask=self.mask,
            surface_embeddings=self.surface_embeddings,
            surface_local_probabilities=self.surface_probabilities,
            mention_ids=self.ids,
            surface_mention_ids=self.ids,
        )
        clean_normalized = (
            self.clean_probabilities
            / self.clean_probabilities.sum(axis=1, keepdims=True)
        )
        surface_normalized = (
            self.surface_probabilities
            / self.surface_probabilities.sum(axis=1, keepdims=True)
        )
        np.testing.assert_allclose(
            view.embeddings[self.mask],
            self.surface_embeddings[self.mask],
        )
        np.testing.assert_allclose(
            view.embeddings[~self.mask],
            self.clean_embeddings[~self.mask],
        )
        np.testing.assert_allclose(
            view.local_probabilities[self.mask],
            surface_normalized[self.mask],
        )
        np.testing.assert_allclose(
            view.local_probabilities[~self.mask],
            clean_normalized[~self.mask],
        )
        np.testing.assert_allclose(
            view.local_probabilities.sum(axis=1),
            np.ones(len(self.mask)),
        )
        probability_start = self.clean_embeddings.shape[1]
        probability_end = probability_start + self.clean_probabilities.shape[1]
        np.testing.assert_array_equal(
            view.aligned_features[:, probability_start:probability_end],
            view.local_probabilities,
        )
        np.testing.assert_array_equal(
            view.aligned_features[:, -1:],
            view.mask_indicator,
        )

    def test_zero_uniform_view_and_identity_are_normalized(self) -> None:
        zero_view = build_evidence_view(
            self.clean_embeddings,
            self.clean_probabilities,
            mode="zero+uniform",
            mask=self.mask,
        )
        np.testing.assert_array_equal(
            zero_view.embeddings[self.mask],
            np.zeros((2, 2), dtype=np.float32),
        )
        np.testing.assert_allclose(
            zero_view.local_probabilities[self.mask],
            np.full((2, 2), 0.5),
        )
        clean_normalized = (
            self.clean_probabilities
            / self.clean_probabilities.sum(axis=1, keepdims=True)
        )
        np.testing.assert_allclose(
            zero_view.embeddings[~self.mask],
            self.clean_embeddings[~self.mask],
        )
        np.testing.assert_allclose(
            zero_view.local_probabilities[~self.mask],
            clean_normalized[~self.mask],
        )
        np.testing.assert_array_equal(
            zero_view.mask_indicator[:, 0],
            self.mask.astype(np.float32),
        )
        np.testing.assert_allclose(
            zero_view.local_probabilities.sum(axis=1),
            np.ones(4),
        )

        identity = build_evidence_view(
            self.clean_embeddings,
            self.clean_probabilities,
            mode="identity",
            mask=self.mask,
        )
        np.testing.assert_array_equal(
            identity.embeddings,
            self.clean_embeddings,
        )
        np.testing.assert_array_equal(
            identity.mask_indicator,
            np.zeros((4, 1), dtype=np.float32),
        )
        np.testing.assert_allclose(
            identity.local_probabilities,
            clean_normalized,
        )
        probability_start = self.clean_embeddings.shape[1]
        probability_end = probability_start + self.clean_probabilities.shape[1]
        np.testing.assert_array_equal(
            identity.aligned_features[:, probability_start:probability_end],
            identity.local_probabilities,
        )

        already_normalized = clean_normalized.astype(np.float32)
        exact_identity = build_evidence_view(
            self.clean_embeddings,
            already_normalized,
            mode="identity",
        )
        np.testing.assert_array_equal(
            exact_identity.local_probabilities,
            already_normalized,
        )

    def test_surface_row_misalignment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not aligned"):
            build_evidence_view(
                self.clean_embeddings,
                self.clean_probabilities,
                mode="surface_only",
                mask=self.mask,
                surface_embeddings=self.surface_embeddings,
                surface_local_probabilities=self.surface_probabilities,
                mention_ids=self.ids,
                surface_mention_ids=["m1", "m0", "m2", "m3"],
            )
        with self.assertRaisesRegex(ValueError, "aligned mention_ids"):
            build_evidence_view(
                self.clean_embeddings,
                self.clean_probabilities,
                mode="surface_only",
                mask=self.mask,
                surface_embeddings=self.surface_embeddings,
                surface_local_probabilities=self.surface_probabilities,
            )


class DeployableCohortTests(unittest.TestCase):
    def test_ambiguity_uses_train_labels_but_not_evaluation_truth(self) -> None:
        train_one_mentions = [
            _mention("t1", "train-1", 0, "Київ", "LOC"),
            _mention("t1b", "train-1", 2, "Київ", "LOC"),
            _mention("t1c", "train-1", 4, "Київ", "LOC"),
            _mention("t2", "train-1", 10, "НБУ", "ORG"),
        ]
        train_two_mentions = [
            _mention("t3", "train-2", 0, "Київ", "ORG"),
            _mention("t3b", "train-2", 2, "Київ", "ORG"),
            _mention("t3c", "train-2", 4, "Київ", "ORG"),
            _mention("t4", "train-2", 10, "НБУ", "ORG"),
        ]
        test_mentions = [
            _mention("e1", "test-1", 0, "Київ", "LOC"),
            _mention("e2", "test-1", 10, "Київ", "LOC"),
            _mention("e3", "test-1", 20, "НБУ", "ORG"),
            _mention("e4", "test-1", 30, "Нове", "MISC"),
        ]
        documents = [
            _document("train-1", "a", "train", train_one_mentions),
            _document("train-2", "b", "train", train_two_mentions),
            _document(
                "test-label-trap",
                "a",
                "test",
                [_mention("leak", "test-label-trap", 0, "Нове", "ORG")],
            ),
        ]
        stats = build_train_surface_label_stats(documents)
        masks = repeat_ambiguity_cohort_masks(test_mentions, stats)

        np.testing.assert_array_equal(
            masks["repeat"],
            np.asarray([True, True, False, False]),
        )
        np.testing.assert_array_equal(
            masks["train_ambiguous"],
            np.asarray([True, True, False, False]),
        )
        np.testing.assert_array_equal(
            masks["train_unseen"],
            np.asarray([False, False, False, True]),
        )

        changed_truth = [
            _mention(
                mention.mention_id,
                mention.doc_id,
                mention.start,
                mention.text,
                "CHANGED",
            )
            for mention in test_mentions
        ]
        changed_masks = repeat_ambiguity_cohort_masks(changed_truth, stats)
        for name in masks:
            np.testing.assert_array_equal(masks[name], changed_masks[name])

    def test_strict_ambiguity_support_and_entropy_boundaries(self) -> None:
        specifications = {
            "Strict": ["A", "A", "A", "B", "B"],
            "LowSupport": ["A", "A", "B", "B"],
            "LowEntropy": ["A"] * 9 + ["B"],
        }
        train_mentions: list[Mention] = []
        mention_index = 0
        for surface, labels in specifications.items():
            for label in labels:
                train_mentions.append(
                    _mention(
                        f"threshold-{mention_index}",
                        "threshold-train",
                        mention_index * 2,
                        surface,
                        label,
                    )
                )
                mention_index += 1
        stats = build_train_surface_label_stats(
            [
                _document(
                    "threshold-train",
                    "a",
                    "train",
                    train_mentions,
                )
            ]
        )
        self.assertAlmostEqual(stats["strict"].entropy_bits, 0.97095, places=4)
        self.assertAlmostEqual(stats["lowsupport"].entropy_bits, 1.0)
        self.assertLess(stats["lowentropy"].entropy_bits, 0.5)

        evaluation_mentions = [
            _mention("s", "threshold-test", 0, "Strict", "IGNORED"),
            _mention("u", "threshold-test", 2, "LowSupport", "IGNORED"),
            _mention("e", "threshold-test", 4, "LowEntropy", "IGNORED"),
        ]
        masks = repeat_ambiguity_cohort_masks(evaluation_mentions, stats)
        np.testing.assert_array_equal(
            masks["train_ambiguous"],
            np.asarray([True, False, False]),
        )


if __name__ == "__main__":
    unittest.main()
