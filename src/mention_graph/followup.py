from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .data import (
    corpus_statistics,
    documents_for_split,
    flatten_mentions,
)
from .encoding import encode_with_cache
from .experiments import (
    _embedding_lookup,
    _labels_array,
    _labels_for_experiment,
    _manifest,
    _split_embeddings,
    config_checksum,
    load_experiment_documents,
    resolve_device,
    source_checksum,
)
from .graph import normalize_surface
from .metrics import (
    classification_metrics,
    paired_document_bootstrap,
    probability_metrics,
    repetition_buckets,
)
from .models import parameter_count
from .relational import (
    RELATION_NAMES,
    RelationalGraphBatch,
    RelationalGraphSource,
    alias_threshold_diagnostics,
    build_relational_source,
    empty_edge_indices,
    materialize_relational_batch,
    relation_overlap_statistics,
    relation_signal_statistics,
    select_alias_threshold,
)
from .reporting import save_json, write_csv
from .schema import Document, Mention
from .training import (
    fit_local_classifier,
    fit_relational_classifier,
    out_of_fold_probabilities,
    predict_local_probabilities,
    predict_relational_probabilities,
)


DEFAULT_ARMS = (
    {
        "model": "N0_node_only",
        "configuration": "none",
        "features": "base",
    },
    {
        "model": "N1_node_stats",
        "configuration": "none",
        "features": "stats",
    },
    {
        "model": "G_repeat",
        "configuration": "repeat",
        "features": "base",
    },
    {
        "model": "G_alias_repeat",
        "configuration": "alias_repeat",
        "features": "base",
    },
    {
        "model": "G_context",
        "configuration": "context",
        "features": "base",
    },
    {
        "model": "G_untyped",
        "configuration": "untyped_all",
        "features": "base",
    },
    {
        "model": "G_untyped_random",
        "configuration": "untyped_random",
        "features": "base",
    },
    {
        "model": "G_typed",
        "configuration": "typed_all",
        "features": "base",
    },
    {
        "model": "G_type_shuffle",
        "configuration": "type_shuffle",
        "features": "base",
    },
    {
        "model": "G_typed_random",
        "configuration": "random",
        "features": "base",
    },
    {
        "model": "G_alias",
        "configuration": "alias_all",
        "features": "base",
    },
)
PRIMARY_CONTRASTS = (
    ("N0_node_only", "G_untyped"),
    ("G_untyped_random", "G_untyped"),
    ("G_untyped", "G_typed"),
    ("G_type_shuffle", "G_typed"),
    ("G_typed_random", "G_typed"),
    ("N0_node_only", "G_repeat"),
    ("G_typed", "G_alias"),
)
EXPLORATORY_CONTRASTS = (
    ("N0_node_only", "G_alias_repeat"),
    ("N0_node_only", "G_alias"),
    ("G_repeat", "G_alias_repeat"),
    ("G_alias_repeat", "G_alias"),
)
ACTIVE_RELATION_BRANCHES = {
    "none": 0,
    "repeat": 1,
    "alias_repeat": 2,
    "context": 2,
    "untyped_all": 1,
    "untyped_random": 1,
    "typed_all": 3,
    "type_shuffle": 3,
    "random": 3,
    "alias_all": 4,
}


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan")
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    )


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _effective_relational_parameter_count(
    total: int,
    active_relation_branches: int,
    model_config: dict[str, object],
) -> int:
    hidden = int(model_config.get("hidden_dimension", 128))
    layers = int(model_config.get("layers", 1))
    parameters_per_branch = hidden * hidden + (2 * hidden + 1)
    inactive = len(RELATION_NAMES) - active_relation_branches
    return total - inactive * layers * parameters_per_branch


def _relational_features(
    embeddings: np.ndarray,
    local_probabilities: np.ndarray,
    source: RelationalGraphSource,
    *,
    feature_set: str,
) -> np.ndarray:
    if not (
        len(embeddings)
        == len(local_probabilities)
        == len(source.structural_features)
    ):
        raise ValueError("relational feature rows are misaligned")
    if feature_set == "base":
        # Length, relative sentence position, and casing are topology-free.
        structural = source.structural_features[:, [0, 1, 3]]
    elif feature_set == "stats":
        # Adds exact-surface frequency and the four semantic degrees.
        structural = source.structural_features
    else:
        raise ValueError(f"unknown relational feature set {feature_set!r}")
    return np.concatenate(
        [embeddings, local_probabilities, structural],
        axis=1,
    ).astype(np.float32)


def _metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    *,
    include_empty_classes: bool = True,
) -> dict[str, object]:
    result = classification_metrics(
        truth,
        probabilities.argmax(axis=1),
        labels,
        include_empty_classes=include_empty_classes,
    )
    result.update(probability_metrics(truth, probabilities))
    return result


def _connected_mask(
    source: RelationalGraphSource,
    relations: Iterable[str],
) -> np.ndarray:
    mask = np.zeros(len(source.labels), dtype=bool)
    for document_id, semantic in source.document_relation_edges.items():
        offset, _ = source.document_offsets[document_id]
        for relation in relations:
            for left, right in semantic[relation]:
                mask[offset + left] = True
                mask[offset + right] = True
    return mask


def _fixed_subgroups(
    source: RelationalGraphSource,
    mentions: list[Mention],
) -> dict[str, np.ndarray]:
    connected = _connected_mask(source, ("sent", "repeat", "near"))
    exact_repeat = _connected_mask(source, ("repeat",))
    alias = _connected_mask(source, ("alias",))
    buckets = repetition_buckets(mentions)
    singleton = buckets == "1"
    repeated = ~singleton
    return {
        "all": np.ones(len(mentions), dtype=bool),
        "base_union_connected": connected,
        "base_union_isolated": ~connected,
        "repeated_2plus": repeated,
        "singleton": singleton,
        "singleton_connected": singleton & connected,
        "singleton_isolated": singleton & ~connected,
        "exact_repeat_connected": exact_repeat,
        "alias_connected": alias,
    }


def _transition_metrics(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    mask: np.ndarray,
    labels: list[str],
) -> dict[str, object]:
    truth = truth[mask]
    baseline_prediction = baseline_prediction[mask]
    candidate_prediction = candidate_prediction[mask]
    baseline_correct_mask = baseline_prediction == truth
    candidate_correct_mask = candidate_prediction == truth
    corrected = int((~baseline_correct_mask & candidate_correct_mask).sum())
    harmed = int((baseline_correct_mask & ~candidate_correct_mask).sum())
    baseline_errors = int((~baseline_correct_mask).sum())
    baseline_correct = int(baseline_correct_mask.sum())
    support = len(truth)
    baseline_metrics = classification_metrics(
        truth,
        baseline_prediction,
        labels,
        include_empty_classes=False,
    )
    candidate_metrics = classification_metrics(
        truth,
        candidate_prediction,
        labels,
        include_empty_classes=False,
    )
    return {
        "support": support,
        "baseline_errors": baseline_errors,
        "corrected": corrected,
        "correction_rate": (
            corrected / baseline_errors if baseline_errors else 0.0
        ),
        "baseline_correct": baseline_correct,
        "harmed": harmed,
        "harm_rate": harmed / baseline_correct if baseline_correct else 0.0,
        "flipped": int((baseline_prediction != candidate_prediction).sum()),
        "flip_rate": (
            float((baseline_prediction != candidate_prediction).mean())
            if support
            else 0.0
        ),
        "net_accuracy_change": (
            (corrected - harmed) / support if support else 0.0
        ),
        "baseline_accuracy": baseline_metrics["accuracy"],
        "candidate_accuracy": candidate_metrics["accuracy"],
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "candidate_macro_f1": candidate_metrics["macro_f1"],
        "delta_macro_f1": (
            float(candidate_metrics["macro_f1"])
            - float(baseline_metrics["macro_f1"])
        ),
    }


def _summarize_models(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["model"])].append(record)
    rows: list[dict[str, object]] = []
    for model, model_records in sorted(grouped.items()):
        row: dict[str, object] = {
            "model": model,
            "runs": len(model_records),
            "configuration": model_records[0]["configuration"],
            "features": model_records[0]["features"],
            "parameter_count": model_records[0]["parameter_count"],
            "active_relation_branches": model_records[0][
                "active_relation_branches"
            ],
            "effective_trainable_parameters": model_records[0][
                "effective_trainable_parameters"
            ],
        }
        for metric in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "micro_f1",
            "negative_log_likelihood",
            "brier_score",
            "ece_10_bins",
        ):
            mean, standard_deviation = _mean_std(
                float(record["metrics"][metric])
                for record in model_records
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = standard_deviation
        rows.append(row)
    baseline = next(
        (
            float(row["macro_f1_mean"])
            for row in rows
            if row["model"] == "N0_node_only"
        ),
        float("nan"),
    )
    for row in rows:
        row["delta_macro_f1_vs_N0"] = (
            float(row["macro_f1_mean"]) - baseline
        )
    return rows


def _summarize_flat_records(
    records: list[dict[str, object]],
    *,
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    rows: list[dict[str, object]] = []
    for group_key, group_records in sorted(
        grouped.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        row = {key: value for key, value in zip(keys, group_key, strict=True)}
        row["runs"] = len(group_records)
        for metric in metrics:
            mean, standard_deviation = _mean_std(
                float(record[metric]) for record in group_records
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = standard_deviation
        rows.append(row)
    return rows


def _aggregate_counterfactual_per_seed(
    records: list[dict[str, object]],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for record in records:
        grouped[
            (
                int(record["seed"]),
                str(record["model"]),
                str(record["intervention"]),
            )
        ].append(record)
    rows: list[dict[str, object]] = []
    for (seed, model, intervention), group_records in sorted(grouped.items()):
        row: dict[str, object] = {
            "seed": seed,
            "model": model,
            "intervention": intervention,
            "topology_permutations": len(group_records),
        }
        for metric in metrics:
            row[metric] = float(
                np.mean([float(record[metric]) for record in group_records])
            )
        rows.append(row)
    return rows


def _holm_adjust(rows: list[dict[str, object]]) -> None:
    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["run_seed"])].append(row)
    for seed_rows in by_seed.values():
        ordered = sorted(
            seed_rows,
            key=lambda row: float(row["two_sided_bootstrap_p"]),
        )
        running = 0.0
        count = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(
                1.0,
                (count - rank) * float(row["two_sided_bootstrap_p"]),
            )
            running = max(running, adjusted)
            row["holm_adjusted_p_within_seed"] = running


def _bootstrap_two_sided_p(values: np.ndarray) -> float:
    non_positive = (np.count_nonzero(values <= 0.0) + 1) / (
        len(values) + 1
    )
    non_negative = (np.count_nonzero(values >= 0.0) + 1) / (
        len(values) + 1
    )
    return float(min(1.0, 2.0 * min(non_positive, non_negative)))


def _nested_seed_document_bootstrap(
    truth: np.ndarray,
    predictions_by_seed: dict[int, dict[str, np.ndarray]],
    document_ids: list[str],
    document_sources: dict[str, str],
    labels: list[str],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    unique_documents = sorted(set(document_ids))
    document_to_index = {
        document_id: index
        for index, document_id in enumerate(unique_documents)
    }
    indices_by_document = {
        document_id: np.flatnonzero(
            np.asarray(document_ids) == document_id
        )
        for document_id in unique_documents
    }
    documents_by_source: dict[str, list[str]] = defaultdict(list)
    for document_id in unique_documents:
        documents_by_source[document_sources[document_id]].append(document_id)
    run_seeds = sorted(predictions_by_seed)
    class_count = len(labels)

    def document_confusions(
        prediction: np.ndarray,
    ) -> np.ndarray:
        matrices = np.zeros(
            (len(unique_documents), class_count, class_count),
            dtype=np.int64,
        )
        for document_id, indices in indices_by_document.items():
            np.add.at(
                matrices[document_to_index[document_id]],
                (truth[indices], prediction[indices]),
                1,
            )
        return matrices

    confusion_cache = {
        run_seed: {
            model: document_confusions(prediction)
            for model, prediction in model_predictions.items()
        }
        for run_seed, model_predictions in predictions_by_seed.items()
    }

    def macro_f1_from_matrix(matrix: np.ndarray) -> float:
        true_positive = np.diag(matrix).astype(np.float64)
        predicted = matrix.sum(axis=0).astype(np.float64)
        support = matrix.sum(axis=1).astype(np.float64)
        precision = np.divide(
            true_positive,
            predicted,
            out=np.zeros_like(true_positive),
            where=predicted > 0,
        )
        recall = np.divide(
            true_positive,
            support,
            out=np.zeros_like(true_positive),
            where=support > 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        return float(f1.mean())

    def batched_macro_f1(matrices: np.ndarray) -> np.ndarray:
        true_positive = np.diagonal(
            matrices,
            axis1=1,
            axis2=2,
        ).astype(np.float64)
        predicted = matrices.sum(axis=1).astype(np.float64)
        support = matrices.sum(axis=2).astype(np.float64)
        precision = np.divide(
            true_positive,
            predicted,
            out=np.zeros_like(true_positive),
            where=predicted > 0,
        )
        recall = np.divide(
            true_positive,
            support,
            out=np.zeros_like(true_positive),
            where=support > 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        return f1.mean(axis=1)

    generator = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    contrasts = [
        ("primary", *contrast) for contrast in PRIMARY_CONTRASTS
    ] + [
        ("exploratory", *contrast) for contrast in EXPLORATORY_CONTRASTS
    ]
    for contrast_index, (family, model_a, model_b) in enumerate(contrasts):
        if not all(
            model_a in predictions_by_seed[run_seed]
            and model_b in predictions_by_seed[run_seed]
            for run_seed in run_seeds
        ):
            continue
        observed_seed_deltas: list[float] = []
        for run_seed in run_seeds:
            observed_seed_deltas.append(
                macro_f1_from_matrix(
                    confusion_cache[run_seed][model_b].sum(axis=0)
                )
                - macro_f1_from_matrix(
                    confusion_cache[run_seed][model_a].sum(axis=0)
                )
            )
        distribution = np.empty(samples, dtype=np.float64)
        contrast_generator = np.random.default_rng(
            generator.integers(0, np.iinfo(np.int64).max)
            + contrast_index
        )
        confusion_a = np.stack(
            [
                confusion_cache[run_seed][model_a]
                for run_seed in run_seeds
            ],
            axis=0,
        )
        confusion_b = np.stack(
            [
                confusion_cache[run_seed][model_b]
                for run_seed in run_seeds
            ],
            axis=0,
        )
        source_index_groups = [
            np.asarray(
                [
                    document_to_index[document_id]
                    for document_id in source_documents
                ],
                dtype=np.int64,
            )
            for source_documents in documents_by_source.values()
        ]
        batch_size = 200
        for batch_left in range(0, samples, batch_size):
            batch_right = min(samples, batch_left + batch_size)
            current_size = batch_right - batch_left
            slot_deltas = np.empty(
                (current_size, len(run_seeds)),
                dtype=np.float64,
            )
            # The same stratified document resample is crossed with every
            # sampled optimization seed because all seeds saw the same test
            # documents.
            sampled_document_indices = np.concatenate(
                [
                    contrast_generator.choice(
                        source_indices,
                        size=(current_size, len(source_indices)),
                        replace=True,
                    )
                    for source_indices in source_index_groups
                ],
                axis=1,
            )
            for slot in range(len(run_seeds)):
                sampled_seed_indices = contrast_generator.integers(
                    0,
                    len(run_seeds),
                    size=current_size,
                )
                selected_a = confusion_a[
                    sampled_seed_indices[:, None],
                    sampled_document_indices,
                ].sum(axis=1)
                selected_b = confusion_b[
                    sampled_seed_indices[:, None],
                    sampled_document_indices,
                ].sum(axis=1)
                slot_deltas[:, slot] = (
                    batched_macro_f1(selected_b)
                    - batched_macro_f1(selected_a)
                )
            distribution[batch_left:batch_right] = slot_deltas.mean(axis=1)
        rows.append(
            {
                "family": family,
                "model_a": model_a,
                "model_b": model_b,
                "optimization_seeds": len(run_seeds),
                "documents": len(unique_documents),
                "samples": samples,
                "observed_mean_delta_macro_f1": float(
                    np.mean(observed_seed_deltas)
                ),
                "seed_deltas_macro_f1": observed_seed_deltas,
                "bootstrap_mean_delta_macro_f1": float(distribution.mean()),
                "confidence_interval_95": [
                    float(value)
                    for value in np.quantile(distribution, [0.025, 0.975])
                ],
                "two_sided_bootstrap_p": _bootstrap_two_sided_p(
                    distribution
                ),
                "resampling": (
                    "optimization seeds, then documents stratified by source"
                ),
            }
        )
    for family in ("primary", "exploratory"):
        family_rows = [row for row in rows if row["family"] == family]
        ordered = sorted(
            family_rows,
            key=lambda row: float(row["two_sided_bootstrap_p"]),
        )
        running = 0.0
        count = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(
                1.0,
                (count - rank) * float(row["two_sided_bootstrap_p"]),
            )
            running = max(running, adjusted)
            row["holm_adjusted_p_within_family"] = running
    return rows


def _alias_examples(
    documents: list[Document],
    source: RelationalGraphSource,
) -> list[dict[str, object]]:
    document_lookup = {document.doc_id: document for document in documents}
    rows: list[dict[str, object]] = []
    for document_id, semantic in source.document_relation_edges.items():
        mentions = document_lookup[document_id].sorted_mentions()
        for left, right in sorted(semantic["alias"]):
            rows.append(
                {
                    "document_id": document_id,
                    "left_mention_id": mentions[left].mention_id,
                    "left_text": mentions[left].text,
                    "left_label": mentions[left].label,
                    "right_mention_id": mentions[right].mention_id,
                    "right_text": mentions[right].text,
                    "right_label": mentions[right].label,
                    "same_gold_label": (
                        mentions[left].label == mentions[right].label
                    ),
                    "character_distance": min(
                        abs(mentions[left].start - mentions[right].end),
                        abs(mentions[right].start - mentions[left].end),
                    ),
                }
            )
    return rows


def _undirected_edges(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(left), int(right))
        for left, right in edge_index.T
        if left < right
    }


def _control_strength_diagnostics(
    source: RelationalGraphSource,
    topology_seeds: list[int],
    control_batches: dict[
        str,
        dict[int, RelationalGraphBatch],
    ],
) -> list[dict[str, object]]:
    typed = materialize_relational_batch(
        source,
        configuration="typed_all",
        seed=0,
    )
    untyped = materialize_relational_batch(
        source,
        configuration="untyped_all",
        seed=0,
    )
    rows: list[dict[str, object]] = []
    for topology_seed in topology_seeds:
        for control, baseline, candidate, relations in (
            (
                "untyped_random",
                untyped,
                control_batches["untyped_random"][topology_seed],
                ("sent",),
            ),
            (
                "typed_random",
                typed,
                control_batches["random"][topology_seed],
                ("sent", "repeat", "near"),
            ),
            (
                "type_shuffle",
                typed,
                control_batches["type_shuffle"][topology_seed],
                ("sent", "repeat", "near"),
            ),
        ):
            channels = list(relations)
            if control == "type_shuffle":
                channels.append("union")
            for relation in channels:
                if relation == "union":
                    original = set().union(
                        *(
                            _undirected_edges(baseline.edge_indices[key])
                            for key in ("sent", "repeat", "near")
                        )
                    )
                    controlled = set().union(
                        *(
                            _undirected_edges(candidate.edge_indices[key])
                            for key in ("sent", "repeat", "near")
                        )
                    )
                    original_targets = np.asarray(
                        [
                            node
                            for left, right in original
                            for node in (left, right)
                        ],
                        dtype=np.int64,
                    )
                    controlled_targets = np.asarray(
                        [
                            node
                            for left, right in controlled
                            for node in (left, right)
                        ],
                        dtype=np.int64,
                    )
                else:
                    original = _undirected_edges(
                        baseline.edge_indices[relation]
                    )
                    controlled = _undirected_edges(
                        candidate.edge_indices[relation]
                    )
                    original_targets = baseline.edge_indices[relation][1]
                    controlled_targets = candidate.edge_indices[relation][1]
                retained = len(original & controlled)
                original_degrees = np.bincount(
                    original_targets,
                    minlength=len(source.labels),
                )
                controlled_degrees = np.bincount(
                    controlled_targets,
                    minlength=len(source.labels),
                )
                unchanged_documents = 0
                for left, right in source.document_offsets.values():
                    original_local = {
                        edge
                        for edge in original
                        if left <= edge[0] < right
                    }
                    controlled_local = {
                        edge
                        for edge in controlled
                        if left <= edge[0] < right
                    }
                    unchanged_documents += int(
                        original_local == controlled_local
                    )
                rows.append(
                    {
                        "control": control,
                        "topology_seed": topology_seed,
                        "relation_channel": (
                            "union" if control == "untyped_random" else relation
                        ),
                        "original_edges": len(original),
                        "controlled_edges": len(controlled),
                        "retained_edges": retained,
                        "retained_fraction": (
                            retained / len(original) if original else 1.0
                        ),
                        "changed_fraction": (
                            1.0 - retained / len(original)
                            if original
                            else 0.0
                        ),
                        "per_node_degree_preserved": bool(
                            np.array_equal(
                                original_degrees,
                                controlled_degrees,
                            )
                        ),
                        "unchanged_documents": unchanged_documents,
                        "documents": len(source.document_offsets),
                    }
                )
    return rows


def _safe_key(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.casefold()).strip("_")


def run_followup_experiment(
    config: dict[str, Any],
    *,
    project_root: Path,
    output_directory: Path,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    checksum = config_checksum(config)
    code_checksum = source_checksum(project_root)
    device = resolve_device(str(config.get("device", "auto")))
    documents = load_experiment_documents(config, project_root)
    labels = _labels_for_experiment(documents, config)
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_documents = documents_for_split(documents, "train")
    validation_documents = documents_for_split(documents, "validation")
    test_documents = documents_for_split(documents, "test")
    followup = config["followup"]
    near_threshold = int(config["graph"].get("near_threshold_tokens", 50))

    alias_candidates = [
        float(value)
        for value in followup.get(
            "alias_threshold_candidates",
            [0.88, 0.90, 0.92, 0.94, 0.96],
        )
    ]
    alias_diagnostics = alias_threshold_diagnostics(
        validation_documents,
        alias_candidates,
    )
    alias_threshold = select_alias_threshold(
        alias_diagnostics,
        minimum_same_label_rate=float(
            followup.get("alias_minimum_same_label_rate", 1.0)
        ),
    )
    for row in alias_diagnostics:
        row["selected"] = bool(
            abs(float(row["threshold"]) - alias_threshold) < 1e-12
        )
        row["selection_split"] = "validation"
    write_csv(output_directory / "table_f1_alias_selection.csv", alias_diagnostics)

    sources = {
        "train": build_relational_source(
            train_documents,
            label_to_id=label_to_id,
            near_threshold_tokens=near_threshold,
            alias_threshold=alias_threshold,
        ),
        "validation": build_relational_source(
            validation_documents,
            label_to_id=label_to_id,
            near_threshold_tokens=near_threshold,
            alias_threshold=alias_threshold,
        ),
        "test": build_relational_source(
            test_documents,
            label_to_id=label_to_id,
            near_threshold_tokens=near_threshold,
            alias_threshold=alias_threshold,
        ),
    }
    edge_signal_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    for split, source in sources.items():
        edge_signal_rows.extend(
            relation_signal_statistics(source, split=split)
        )
        overlap_rows.extend(
            relation_overlap_statistics(source, split=split)
        )
    write_csv(output_directory / "table_f2_edge_signal.csv", edge_signal_rows)
    write_csv(output_directory / "edge_overlap.csv", overlap_rows)
    save_json(
        output_directory / "alias_validation_examples.json",
        _alias_examples(validation_documents, sources["validation"]),
    )
    save_json(
        output_directory / "alias_test_examples.json",
        _alias_examples(test_documents, sources["test"]),
    )

    cache_root = project_root / str(
        config.get("embedding_cache", "data/cache/embeddings")
    )
    model_cache = project_root / str(config.get("model_cache", "data/models"))
    all_ids, all_embeddings, embedding_cache_path = encode_with_cache(
        documents,
        encoder=config["encoder"],
        cache_root=cache_root,
        model_cache=model_cache,
        device=device,
    )
    lookup = _embedding_lookup(all_ids, all_embeddings)
    train_mentions, train_embeddings = _split_embeddings(train_documents, lookup)
    validation_mentions, validation_embeddings = _split_embeddings(
        validation_documents,
        lookup,
    )
    test_mentions, test_embeddings = _split_embeddings(test_documents, lookup)
    for split, mentions in (
        ("train", train_mentions),
        ("validation", validation_mentions),
        ("test", test_mentions),
    ):
        if sources[split].mention_ids != [
            mention.mention_id for mention in mentions
        ]:
            raise RuntimeError(f"{split} relational graph is misaligned")
    train_truth = _labels_array(train_mentions, label_to_id)
    validation_truth = _labels_array(validation_mentions, label_to_id)
    test_truth = _labels_array(test_mentions, label_to_id)
    document_sources = {document.doc_id: document.source for document in documents}
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    local_config = config["training"]["local"]
    model_config = followup["model"]
    oof_folds = int(config["training"].get("oof_folds", 5))
    arms = [
        {
            "model": str(arm["model"]),
            "configuration": str(arm["configuration"]),
            "features": str(arm["features"]),
        }
        for arm in followup.get("arms", DEFAULT_ARMS)
    ]

    model_records: list[dict[str, object]] = []
    subgroup_records: list[dict[str, object]] = []
    transition_records: list[dict[str, object]] = []
    counterfactual_records: list[dict[str, object]] = []
    bootstrap_records: list[dict[str, object]] = []
    graph_diagnostics: list[dict[str, object]] = []
    predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}
    test_groups = _fixed_subgroups(sources["test"], test_mentions)
    bootstrap_samples = int(
        config.get("statistics", {}).get("bootstrap_samples", 10000)
    )
    bootstrap_seed = int(
        config.get("statistics", {}).get("bootstrap_seed", 8675309)
    )
    control_topology_seeds = [
        int(value)
        for value in followup.get(
            "control_topology_seeds",
            [101, 211, 307, 401, 503],
        )
    ]
    test_control_batches = {
        configuration: {
            topology_seed: materialize_relational_batch(
                sources["test"],
                configuration=configuration,
                seed=topology_seed,
            )
            for topology_seed in control_topology_seeds
        }
        for configuration in ("untyped_random", "random", "type_shuffle")
    }
    write_csv(
        output_directory / "table_f9_control_strength.csv",
        _control_strength_diagnostics(
            sources["test"],
            control_topology_seeds,
            test_control_batches,
        ),
    )

    for seed in seeds:
        seed_directory = output_directory / f"seed-{seed}"
        seed_directory.mkdir(parents=True, exist_ok=True)
        train_document_ids = [mention.doc_id for mention in train_mentions]
        oof_probabilities, fold_assignments = out_of_fold_probabilities(
            train_embeddings,
            train_truth,
            train_document_ids,
            document_sources,
            class_count=len(labels),
            fold_count=oof_folds,
            config=local_config,
            seed=seed,
            device=device,
        )
        local_model = fit_local_classifier(
            train_embeddings,
            train_truth,
            class_count=len(labels),
            config=local_config,
            seed=seed,
            device=device,
        )
        validation_local = predict_local_probabilities(
            local_model,
            validation_embeddings,
            device=device,
        )
        test_local = predict_local_probabilities(
            local_model,
            test_embeddings,
            device=device,
        )
        local_metrics = _metrics(test_truth, test_local, labels)
        model_records.append(
            {
                "seed": seed,
                "model": "local",
                "configuration": "none",
                "features": "embedding",
                "parameter_count": parameter_count(local_model),
                "active_relation_branches": 0,
                "effective_trainable_parameters": parameter_count(local_model),
                "metrics": local_metrics,
            }
        )
        probabilities_by_model: dict[str, np.ndarray] = {"local": test_local}
        results_by_model: dict[str, object] = {}
        test_batches_by_configuration: dict[str, RelationalGraphBatch] = {}
        base_parameter_count: int | None = None

        for arm in arms:
            model_name = arm["model"]
            configuration = arm["configuration"]
            feature_set = arm["features"]
            train_batch = materialize_relational_batch(
                sources["train"],
                configuration=configuration,
                seed=seed,
            )
            validation_batch = materialize_relational_batch(
                sources["validation"],
                configuration=configuration,
                seed=seed,
            )
            test_batch = materialize_relational_batch(
                sources["test"],
                configuration=configuration,
                seed=seed,
            )
            test_batches_by_configuration[configuration] = test_batch
            train_features = _relational_features(
                train_embeddings,
                oof_probabilities,
                sources["train"],
                feature_set=feature_set,
            )
            validation_features = _relational_features(
                validation_embeddings,
                validation_local,
                sources["validation"],
                feature_set=feature_set,
            )
            test_features = _relational_features(
                test_embeddings,
                test_local,
                sources["test"],
                feature_set=feature_set,
            )
            result = fit_relational_classifier(
                train_features,
                train_truth,
                oof_probabilities,
                train_batch.edge_indices,
                validation_features,
                validation_truth,
                validation_local,
                validation_batch.edge_indices,
                labels=labels,
                relation_names=list(RELATION_NAMES),
                config=model_config,
                seed=seed,
                device=device,
            )
            if feature_set == "base":
                if base_parameter_count is None:
                    base_parameter_count = result.parameter_count
                elif result.parameter_count != base_parameter_count:
                    raise RuntimeError(
                        "primary relational arms are not parameter-matched"
                    )
            probabilities = predict_relational_probabilities(
                result,
                test_features,
                test_local,
                test_batch.edge_indices,
                list(RELATION_NAMES),
                device=device,
            )
            metrics = _metrics(test_truth, probabilities, labels)
            active_branches = ACTIVE_RELATION_BRANCHES[configuration]
            effective_parameters = _effective_relational_parameter_count(
                result.parameter_count,
                active_branches,
                model_config,
            )
            model_records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "configuration": configuration,
                    "features": feature_set,
                    "parameter_count": result.parameter_count,
                    "active_relation_branches": active_branches,
                    "effective_trainable_parameters": effective_parameters,
                    "metrics": metrics,
                }
            )
            probabilities_by_model[model_name] = probabilities
            results_by_model[model_name] = result
            graph_diagnostics.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "configuration": configuration,
                    "features": feature_set,
                    "parameter_count": result.parameter_count,
                    "active_relation_branches": active_branches,
                    "effective_trainable_parameters": effective_parameters,
                    "best_epoch": result.best_epoch,
                    "best_validation_macro_f1": (
                        result.best_validation_macro_f1
                    ),
                    "train_edge_counts": train_batch.edge_counts,
                    "test_edge_counts": test_batch.edge_counts,
                    "train_isolated_nodes": train_batch.isolated_nodes,
                    "test_isolated_nodes": test_batch.isolated_nodes,
                }
            )
            torch.save(
                {
                    "state_dict": _cpu_state_dict(result.model),
                    "input_dimension": train_features.shape[1],
                    "labels": labels,
                    "relations": list(RELATION_NAMES),
                    "model_config": model_config,
                    "arm": arm,
                    "seed": seed,
                    "config_checksum": checksum,
                    "source_checksum": code_checksum,
                    "best_epoch": result.best_epoch,
                    "best_validation_macro_f1": (
                        result.best_validation_macro_f1
                    ),
                },
                seed_directory / f"{_safe_key(model_name)}_model.pt",
            )

        for model_name, probabilities in probabilities_by_model.items():
            prediction = probabilities.argmax(axis=1)
            for group, mask in test_groups.items():
                if not mask.any():
                    continue
                group_metrics = classification_metrics(
                    test_truth[mask],
                    prediction[mask],
                    labels,
                    include_empty_classes=False,
                )
                subgroup_records.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "subgroup": group,
                        "support": int(mask.sum()),
                        "accuracy": group_metrics["accuracy"],
                        "macro_f1": group_metrics["macro_f1"],
                    }
                )

        transition_contrasts = [
            ("local", "N0_node_only"),
            *PRIMARY_CONTRASTS,
            ("N0_node_only", "G_typed"),
            ("N0_node_only", "G_alias"),
        ]
        for baseline_name, candidate_name in transition_contrasts:
            if (
                baseline_name not in probabilities_by_model
                or candidate_name not in probabilities_by_model
            ):
                continue
            baseline_prediction = probabilities_by_model[
                baseline_name
            ].argmax(axis=1)
            candidate_prediction = probabilities_by_model[
                candidate_name
            ].argmax(axis=1)
            for group, mask in test_groups.items():
                transition_records.append(
                    {
                        "seed": seed,
                        "baseline": baseline_name,
                        "candidate": candidate_name,
                        "subgroup": group,
                        **_transition_metrics(
                            test_truth,
                            baseline_prediction,
                            candidate_prediction,
                            mask,
                            labels,
                        ),
                    }
                )

        # Same-checkpoint interventions isolate message use from retraining.
        for model_name, full_configuration, interventions in (
            (
                "G_untyped",
                "untyped_all",
                ("no_edges", "untyped_random"),
            ),
            (
                "G_typed",
                "typed_all",
                ("no_edges", "type_shuffle", "random", "drop_sent", "drop_repeat", "drop_near"),
            ),
            (
                "G_alias",
                "alias_all",
                ("no_edges", "drop_sent", "drop_repeat", "drop_near", "drop_alias"),
            ),
        ):
            if model_name not in results_by_model:
                continue
            result = results_by_model[model_name]
            full_batch = test_batches_by_configuration[full_configuration]
            full_features = _relational_features(
                test_embeddings,
                test_local,
                sources["test"],
                feature_set="base",
            )
            full_probabilities = probabilities_by_model[model_name]
            full_macro = float(
                classification_metrics(
                    test_truth,
                    full_probabilities.argmax(axis=1),
                    labels,
                )["macro_f1"]
            )
            for intervention in interventions:
                topology_seeds: list[int | None] = (
                    control_topology_seeds
                    if intervention
                    in {"type_shuffle", "random", "untyped_random"}
                    else [None]
                )
                for topology_seed in topology_seeds:
                    if intervention == "no_edges":
                        intervened_edges = empty_edge_indices()
                    elif intervention in {
                        "type_shuffle",
                        "random",
                        "untyped_random",
                    }:
                        intervened_edges = test_control_batches[
                            intervention
                        ][int(topology_seed)].edge_indices
                    elif intervention.startswith("drop_"):
                        relation = intervention.removeprefix("drop_")
                        intervened_edges = {
                            key: (
                                np.empty((2, 0), dtype=np.int64)
                                if key == relation
                                else value
                            )
                            for key, value in full_batch.edge_indices.items()
                        }
                    else:
                        raise ValueError(
                            f"unknown intervention {intervention!r}"
                        )
                    intervened = predict_relational_probabilities(
                        result,
                        full_features,
                        test_local,
                        intervened_edges,
                        list(RELATION_NAMES),
                        device=device,
                    )
                    intervened_metrics = _metrics(
                        test_truth,
                        intervened,
                        labels,
                    )
                    counterfactual_records.append(
                        {
                            "seed": seed,
                            "topology_seed": topology_seed,
                            "model": model_name,
                            "intervention": intervention,
                            "macro_f1": intervened_metrics["macro_f1"],
                            "accuracy": intervened_metrics["accuracy"],
                            "full_macro_f1": full_macro,
                            "full_minus_intervention_macro_f1": (
                                full_macro
                                - float(intervened_metrics["macro_f1"])
                            ),
                            **{
                                f"transition_{key}": value
                                for key, value in _transition_metrics(
                                    test_truth,
                                    intervened.argmax(axis=1),
                                    full_probabilities.argmax(axis=1),
                                    test_groups["all"],
                                    labels,
                                ).items()
                            },
                        }
                    )
                    if (
                        intervention == "no_edges"
                        and model_name in {"G_typed", "G_untyped"}
                    ):
                        isolated = test_groups["base_union_isolated"]
                        if isolated.any() and not np.allclose(
                            full_probabilities[isolated],
                            intervened[isolated],
                            atol=1e-6,
                            rtol=1e-6,
                        ):
                            raise RuntimeError(
                                "isolated-node probabilities changed after edge masking"
                            )

        document_ids = [mention.doc_id for mention in test_mentions]
        for comparison_index, (baseline_name, candidate_name) in enumerate(
            PRIMARY_CONTRASTS
        ):
            if (
                baseline_name not in probabilities_by_model
                or candidate_name not in probabilities_by_model
            ):
                continue
            bootstrap = paired_document_bootstrap(
                test_truth,
                probabilities_by_model[baseline_name].argmax(axis=1),
                probabilities_by_model[candidate_name].argmax(axis=1),
                document_ids,
                labels,
                samples=bootstrap_samples,
                seed=bootstrap_seed + seed * 100 + comparison_index,
            )
            bootstrap.update(
                {
                    "model_a": baseline_name,
                    "model_b": candidate_name,
                    "run_seed": seed,
                    "family": "followup_primary",
                }
            )
            bootstrap_records.append(bootstrap)

        prediction_payload: dict[str, object] = {
            "mention_ids": np.asarray(
                [mention.mention_id for mention in test_mentions]
            ),
            "document_ids": np.asarray(
                [mention.doc_id for mention in test_mentions]
            ),
            "truth": test_truth,
            "config_checksum": np.asarray([checksum]),
            "source_checksum": np.asarray([code_checksum]),
            "oof_fold_assignments": np.asarray(
                [
                    fold_assignments[mention.doc_id]
                    for mention in train_mentions
                ],
                dtype=np.int64,
            ),
        }
        for model_name, probabilities in probabilities_by_model.items():
            prediction_payload[
                f"{_safe_key(model_name)}_probabilities"
            ] = probabilities
        np.savez_compressed(
            seed_directory / "test_predictions.npz",
            **prediction_payload,
        )
        predictions_by_seed[seed] = {
            model_name: probabilities.argmax(axis=1).astype(np.int16)
            for model_name, probabilities in probabilities_by_model.items()
        }
        del local_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    _holm_adjust(bootstrap_records)
    nested_bootstraps = _nested_seed_document_bootstrap(
        test_truth,
        predictions_by_seed,
        [mention.doc_id for mention in test_mentions],
        document_sources,
        labels,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 99173,
    )
    model_summary = _summarize_models(model_records)
    write_csv(output_directory / "table_f3_model_comparison.csv", model_summary)
    subgroup_summary = _summarize_flat_records(
        subgroup_records,
        keys=("model", "subgroup", "support"),
        metrics=("accuracy", "macro_f1"),
    )
    write_csv(output_directory / "table_f4_subgroups.csv", subgroup_summary)
    write_csv(output_directory / "subgroups_by_seed.csv", subgroup_records)
    transition_summary = _summarize_flat_records(
        transition_records,
        keys=("baseline", "candidate", "subgroup", "support"),
        metrics=(
            "baseline_errors",
            "corrected",
            "correction_rate",
            "baseline_correct",
            "harmed",
            "harm_rate",
            "flip_rate",
            "net_accuracy_change",
            "baseline_macro_f1",
            "candidate_macro_f1",
            "delta_macro_f1",
        ),
    )
    write_csv(
        output_directory / "table_f5_correction_harm.csv",
        transition_summary,
    )
    write_csv(
        output_directory / "correction_harm_by_seed.csv",
        transition_records,
    )
    counterfactual_metrics = (
        "macro_f1",
        "accuracy",
        "full_macro_f1",
        "full_minus_intervention_macro_f1",
        "transition_corrected",
        "transition_harmed",
        "transition_net_accuracy_change",
        "transition_delta_macro_f1",
    )
    counterfactual_per_seed = _aggregate_counterfactual_per_seed(
        counterfactual_records,
        counterfactual_metrics,
    )
    counterfactual_summary = _summarize_flat_records(
        counterfactual_per_seed,
        keys=("model", "intervention"),
        metrics=counterfactual_metrics,
    )
    topology_counts = {
        (str(row["model"]), str(row["intervention"])): int(
            row["topology_permutations"]
        )
        for row in counterfactual_per_seed
    }
    for row in counterfactual_summary:
        row["optimization_seeds"] = row.pop("runs")
        row["topology_permutations_per_seed"] = topology_counts[
            (str(row["model"]), str(row["intervention"]))
        ]
    write_csv(
        output_directory / "table_f6_counterfactual.csv",
        counterfactual_summary,
    )
    write_csv(
        output_directory / "counterfactual_by_seed.csv",
        counterfactual_per_seed,
    )
    write_csv(
        output_directory / "counterfactual_raw.csv",
        counterfactual_records,
    )
    write_csv(output_directory / "table_f7_bootstrap.csv", bootstrap_records)
    write_csv(
        output_directory / "table_f7b_nested_bootstrap.csv",
        nested_bootstraps,
    )
    write_csv(
        output_directory / "table_f8_graph_diagnostics.csv",
        graph_diagnostics,
    )
    save_json(output_directory / "metrics_detailed.json", model_records)
    save_json(output_directory / "paired_bootstrap.json", bootstrap_records)
    save_json(
        output_directory / "nested_seed_document_bootstrap.json",
        nested_bootstraps,
    )
    save_json(output_directory / "graph_diagnostics.json", graph_diagnostics)
    save_json(output_directory / "config.json", config)
    save_json(
        output_directory / "dataset_statistics.json",
        corpus_statistics(documents),
    )
    save_json(
        output_directory / "run_manifest.json",
        _manifest(
            config,
            device=device,
            checksum=checksum,
            code_checksum=code_checksum,
            embedding_cache=embedding_cache_path,
        ),
    )
    summary = {
        "output_directory": str(output_directory),
        "alias_threshold": alias_threshold,
        "seeds": seeds,
        "device": device,
        "models": len(model_summary),
        "config_checksum": checksum,
        "source_checksum": code_checksum,
    }
    save_json(output_directory / "summary.json", summary)
    return summary
