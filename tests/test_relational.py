from __future__ import annotations

import unittest

import numpy as np
import torch

from mention_graph.followup import _nested_seed_document_bootstrap
from mention_graph.models import RelationalGatedClassifier, parameter_count
from mention_graph.relational import (
    RELATION_NAMES,
    build_relational_source,
    degree_preserving_edge_rewire,
    materialize_relational_batch,
    surfaces_are_aliases,
)
from mention_graph.synthetic import SYNTHETIC_LABELS, generate_synthetic_corpus


class AliasRuleTests(unittest.TestCase):
    def test_exact_surface_is_not_an_alias_edge(self) -> None:
        self.assertFalse(
            surfaces_are_aliases(
                "Україна",
                "україна",
                token_similarity_threshold=0.9,
            )
        )

    def test_conservative_spelling_variant_is_linked(self) -> None:
        self.assertTrue(
            surfaces_are_aliases(
                "Гончарука",
                "Гончарук",
                token_similarity_threshold=0.9,
            )
        )

    def test_strict_acronym_is_linked(self) -> None:
        self.assertTrue(
            surfaces_are_aliases(
                "НБУ",
                "Національний банк України",
                token_similarity_threshold=0.9,
            )
        )

    def test_numeric_forms_are_not_fuzzy_aliases(self) -> None:
        self.assertFalse(
            surfaces_are_aliases(
                "2025 року",
                "2026 року",
                token_similarity_threshold=0.9,
            )
        )


class RelationalGraphTests(unittest.TestCase):
    @staticmethod
    def _degrees(
        edges: set[tuple[int, int]],
        node_count: int,
    ) -> list[int]:
        result = [0] * node_count
        for left, right in edges:
            result[left] += 1
            result[right] += 1
        return result

    def test_rewire_preserves_each_node_degree(self) -> None:
        edges = {
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 5),
        }
        rewired = degree_preserving_edge_rewire(edges, 6, seed=7)
        self.assertEqual(
            self._degrees(edges, 6),
            self._degrees(rewired, 6),
        )
        self.assertTrue(all(left != right for left, right in rewired))

    def test_structural_features_are_configuration_invariant(self) -> None:
        documents = generate_synthetic_corpus(
            train_documents=4,
            validation_documents=0,
            test_documents=0,
            seed=21,
        )
        source = build_relational_source(
            documents,
            label_to_id={
                label: index
                for index, label in enumerate(SYNTHETIC_LABELS)
            },
            near_threshold_tokens=30,
            alias_threshold=0.9,
        )
        none = materialize_relational_batch(
            source,
            configuration="none",
            seed=3,
        )
        typed = materialize_relational_batch(
            source,
            configuration="typed_all",
            seed=3,
        )
        np.testing.assert_array_equal(
            none.structural_features,
            typed.structural_features,
        )

    def test_type_shuffle_preserves_union_and_relation_counts(self) -> None:
        documents = generate_synthetic_corpus(
            train_documents=5,
            validation_documents=0,
            test_documents=0,
            seed=22,
        )
        source = build_relational_source(
            documents,
            label_to_id={
                label: index
                for index, label in enumerate(SYNTHETIC_LABELS)
            },
            near_threshold_tokens=30,
            alias_threshold=0.9,
        )
        typed = materialize_relational_batch(
            source,
            configuration="typed_all",
            seed=4,
        )
        shuffled = materialize_relational_batch(
            source,
            configuration="type_shuffle",
            seed=4,
        )
        for relation in ("sent", "repeat", "near"):
            self.assertEqual(
                typed.edge_counts[relation],
                shuffled.edge_counts[relation],
            )

        def union(batch: object) -> set[tuple[int, int]]:
            result: set[tuple[int, int]] = set()
            for relation in ("sent", "repeat", "near"):
                edges = batch.edge_indices[relation]
                for source_node, target_node in edges.T:
                    if source_node < target_node:
                        result.add((int(source_node), int(target_node)))
            return result

        self.assertEqual(union(typed), union(shuffled))

        untyped = materialize_relational_batch(
            source,
            configuration="untyped_all",
            seed=4,
        )
        untyped_random = materialize_relational_batch(
            source,
            configuration="untyped_random",
            seed=4,
        )

        def directed_degrees(batch: object) -> np.ndarray:
            targets = batch.edge_indices["sent"][1]
            return np.bincount(
                targets,
                minlength=len(batch.labels),
            )

        np.testing.assert_array_equal(
            directed_degrees(untyped),
            directed_degrees(untyped_random),
        )


class RelationalModelTests(unittest.TestCase):
    def test_empty_edges_are_finite_and_relation_parameters_are_inert(self) -> None:
        torch.manual_seed(3)
        model = RelationalGatedClassifier(
            input_dimension=7,
            hidden_dimension=8,
            class_count=3,
            relation_names=list(RELATION_NAMES),
            layers=1,
            dropout=0.0,
            layer_normalization=True,
        )
        features = torch.randn(5, 7)
        local = torch.softmax(torch.randn(5, 3), dim=1)
        empty = {
            relation: torch.empty((2, 0), dtype=torch.long)
            for relation in RELATION_NAMES
        }
        model.eval()
        before = model(features, empty, local)
        with torch.no_grad():
            for projection in model.layers[0].neighbor_projections.values():
                projection.weight.add_(100.0)
        after = model(features, empty, local)
        self.assertTrue(torch.isfinite(before).all())
        torch.testing.assert_close(before, after)

    def test_parameter_count_does_not_depend_on_adjacency(self) -> None:
        first = RelationalGatedClassifier(
            10,
            12,
            4,
            list(RELATION_NAMES),
            layers=1,
            dropout=0.0,
            layer_normalization=True,
        )
        second = RelationalGatedClassifier(
            10,
            12,
            4,
            list(RELATION_NAMES),
            layers=1,
            dropout=0.0,
            layer_normalization=True,
        )
        self.assertEqual(parameter_count(first), parameter_count(second))


class FollowupStatisticsTests(unittest.TestCase):
    def test_nested_bootstrap_resamples_seeds_and_documents(self) -> None:
        truth = np.asarray([0, 1, 0, 1])
        predictions = {
            1: {
                "N0_node_only": np.asarray([0, 0, 1, 1]),
                "G_untyped": np.asarray([0, 1, 0, 1]),
                "G_random": np.asarray([0, 0, 1, 1]),
                "G_typed": np.asarray([0, 1, 0, 1]),
                "G_type_shuffle": np.asarray([0, 0, 1, 1]),
                "G_repeat": np.asarray([0, 1, 0, 1]),
                "G_alias": np.asarray([0, 1, 0, 1]),
                "G_alias_repeat": np.asarray([0, 1, 0, 1]),
            },
            2: {
                "N0_node_only": np.asarray([0, 0, 1, 1]),
                "G_untyped": np.asarray([0, 1, 0, 1]),
                "G_random": np.asarray([0, 0, 1, 1]),
                "G_typed": np.asarray([0, 1, 0, 1]),
                "G_type_shuffle": np.asarray([0, 0, 1, 1]),
                "G_repeat": np.asarray([0, 1, 0, 1]),
                "G_alias": np.asarray([0, 1, 0, 1]),
                "G_alias_repeat": np.asarray([0, 1, 0, 1]),
            },
        }
        rows = _nested_seed_document_bootstrap(
            truth,
            predictions,
            ["d1", "d1", "d2", "d2"],
            {"d1": "a", "d2": "b"},
            ["A", "B"],
            samples=50,
            seed=9,
        )
        target = next(
            row
            for row in rows
            if row["model_a"] == "N0_node_only"
            and row["model_b"] == "G_untyped"
        )
        self.assertGreater(target["observed_mean_delta_macro_f1"], 0)
        self.assertEqual(target["optimization_seeds"], 2)


if __name__ == "__main__":
    unittest.main()
