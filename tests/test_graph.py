from __future__ import annotations

import unittest

import numpy as np

from mention_graph.graph import (
    build_edge_sets,
    build_graph_batch,
    normalize_surface,
    randomize_edges_by_node_permutation,
)
from mention_graph.synthetic import SYNTHETIC_LABELS, generate_synthetic_corpus


class GraphConstructionTests(unittest.TestCase):
    def test_normalization_is_deterministic(self) -> None:
        self.assertEqual(
            normalize_surface("  «НБУ»  "),
            normalize_surface('"нбу"'),
        )

    def test_repeated_mentions_create_repeat_edges(self) -> None:
        document = generate_synthetic_corpus(
            train_documents=1,
            validation_documents=0,
            test_documents=0,
            seed=12,
        )[0]
        edges = build_edge_sets(document, near_threshold_tokens=10)
        self.assertGreater(len(edges["repeat"]), 0)
        for left, right in edges["repeat"]:
            mentions = document.sorted_mentions()
            self.assertEqual(
                normalize_surface(mentions[left].text),
                normalize_surface(mentions[right].text),
            )

    def test_random_control_preserves_degree_multiset(self) -> None:
        edges = {(0, 1), (0, 2), (2, 3), (3, 4)}
        randomized = randomize_edges_by_node_permutation(edges, 5, seed=4)

        def degrees(edge_set: set[tuple[int, int]]) -> list[int]:
            values = [0] * 5
            for left, right in edge_set:
                values[left] += 1
                values[right] += 1
            return sorted(values)

        self.assertEqual(degrees(edges), degrees(randomized))
        self.assertEqual(len(edges), len(randomized))

    def test_graph_batch_has_no_cross_document_edges(self) -> None:
        documents = generate_synthetic_corpus(
            train_documents=3,
            validation_documents=0,
            test_documents=0,
            seed=13,
        )
        label_to_id = {
            label: index for index, label in enumerate(SYNTHETIC_LABELS)
        }
        batch = build_graph_batch(
            documents,
            label_to_id=label_to_id,
            configuration="all",
            near_threshold_tokens=30,
            seed=1,
        )
        doc_ids = np.asarray(batch.document_ids)
        for source, target in batch.edge_index.T:
            self.assertEqual(doc_ids[source], doc_ids[target])


if __name__ == "__main__":
    unittest.main()

