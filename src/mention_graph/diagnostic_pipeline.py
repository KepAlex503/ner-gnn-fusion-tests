from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .data import corpus_statistics, documents_for_split, flatten_mentions
from .diagnostic_controls import (
    as_single_relation_edge_indices,
    build_sparse_gold_oracle_edge_indices,
    degree_preserving_randomize_edge_indices_by_document,
    semantic_union_pruned_to_same_label_edge_indices,
    stable_nested_document_subsets,
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
from .metrics import classification_metrics, probability_metrics
from .models import parameter_count
from .relational import (
    RELATION_NAMES,
    RelationalGraphSource,
    build_relational_source,
    empty_edge_indices,
    materialize_relational_batch,
)
from .reporting import save_json, write_csv
from .schema import Document, Mention
from .training import (
    document_folds,
    fit_local_classifier,
    fit_relational_classifier,
    predict_local_probabilities,
    predict_relational_probabilities,
)


DIAGNOSTIC_SEEDS = (17, 29, 43, 59, 71, 89, 101, 113, 127, 149)
DEFAULT_CORRUPTION_SEEDS = (101, 211, 307, 401, 503)
DEFAULT_RESOURCE_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
PROPAGATION_ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
SINGLE_RELATION = "sent"


@dataclass(slots=True)
class LocalViews:
    train_mentions: list[Mention]
    train_source: RelationalGraphSource
    train_truth: np.ndarray
    train_clean_embeddings: np.ndarray
    train_surface_embeddings: np.ndarray
    train_clean_probabilities: np.ndarray
    train_surface_probabilities: np.ndarray
    validation_clean_probabilities: np.ndarray
    validation_surface_probabilities: np.ndarray
    test_clean_probabilities: np.ndarray
    test_surface_probabilities: np.ndarray
    train_prior: np.ndarray
    local_model: torch.nn.Module
    fold_assignments: dict[str, int]


@dataclass(slots=True)
class EvidenceInputs:
    features: np.ndarray
    local_probabilities: np.ndarray
    evidence_mask: np.ndarray
    evaluation_mask: np.ndarray
    mode: str


def _stable_digest(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def _repeat_masks(
    mentions: list[Mention],
    *,
    target_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, mention in enumerate(mentions):
        groups[(mention.doc_id, normalize_surface(mention.text))].append(index)
    target = np.zeros(len(mentions), dtype=bool)
    component = np.zeros(len(mentions), dtype=bool)
    repeated_groups = 0
    conflicting_groups = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        repeated_groups += 1
        component[indices] = True
        chosen = min(
            indices,
            key=lambda index: _stable_digest(
                target_seed,
                mentions[index].mention_id,
            ),
        )
        target[chosen] = True
        conflicting_groups += int(
            len({mentions[index].label for index in indices}) > 1
        )
    return (
        target,
        component,
        {
            "groups": repeated_groups,
            "targets": int(target.sum()),
            "component_mentions": int(component.sum()),
            "gold_conflicting_groups_diagnostic_only": conflicting_groups,
        },
    )


def _single_channel_edges(
    edges: np.ndarray,
    *,
    relation: str = SINGLE_RELATION,
) -> dict[str, np.ndarray]:
    result = empty_edge_indices()
    result[relation] = np.asarray(edges, dtype=np.int64).copy()
    return result


def _union_edge_index(
    edge_indices: dict[str, np.ndarray],
    relations: Iterable[str],
) -> np.ndarray:
    undirected: set[tuple[int, int]] = set()
    for relation in relations:
        for left, right in edge_indices[relation].T:
            left_value = int(left)
            right_value = int(right)
            if left_value == right_value:
                continue
            undirected.add(
                (
                    min(left_value, right_value),
                    max(left_value, right_value),
                )
            )
    directed = [
        directed_pair
        for left, right in sorted(undirected)
        for directed_pair in ((left, right), (right, left))
    ]
    return (
        np.asarray(directed, dtype=np.int64).T
        if directed
        else np.empty((2, 0), dtype=np.int64)
    )


def _base_structural_features(source: RelationalGraphSource) -> np.ndarray:
    return source.structural_features[:, [0, 1, 3]].astype(np.float32)


def _evidence_inputs(
    clean_embeddings: np.ndarray,
    surface_embeddings: np.ndarray,
    clean_probabilities: np.ndarray,
    surface_probabilities: np.ndarray,
    source: RelationalGraphSource,
    *,
    mode: str,
    evidence_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    train_prior: np.ndarray,
    degradation_indicator: bool = True,
) -> EvidenceInputs:
    mask = np.asarray(evidence_mask, dtype=bool)
    evaluation = np.asarray(evaluation_mask, dtype=bool)
    if not (
        len(clean_embeddings)
        == len(surface_embeddings)
        == len(clean_probabilities)
        == len(surface_probabilities)
        == len(source.labels)
        == len(mask)
        == len(evaluation)
    ):
        raise ValueError("evidence inputs are not row-aligned")
    embeddings = np.asarray(clean_embeddings, dtype=np.float32).copy()
    probabilities = np.asarray(clean_probabilities, dtype=np.float32).copy()
    if mode == "identity":
        if mask.any():
            raise ValueError("identity evidence must have an empty mask")
    elif mode == "surface":
        embeddings[mask] = surface_embeddings[mask]
        probabilities[mask] = surface_probabilities[mask]
    elif mode == "prior":
        embeddings[mask] = 0.0
        probabilities[mask] = np.asarray(train_prior, dtype=np.float32)
    else:
        raise ValueError(f"unknown evidence mode {mode!r}")
    if not np.isfinite(probabilities).all():
        raise ValueError("evidence probabilities contain non-finite values")
    probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    structural = _base_structural_features(source)
    indicator = (
        mask.astype(np.float32)[:, None]
        if degradation_indicator
        else np.zeros((len(mask), 1), dtype=np.float32)
    )
    features = np.concatenate(
        [
            embeddings,
            probabilities,
            structural,
            indicator,
        ],
        axis=1,
    ).astype(np.float32)
    class_count = probabilities.shape[1]
    probability_slice = features[
        :,
        embeddings.shape[1] : embeddings.shape[1] + class_count,
    ]
    if not np.array_equal(probability_slice, probabilities):
        raise RuntimeError("feature and residual probability channels differ")
    return EvidenceInputs(
        features=features,
        local_probabilities=probabilities,
        evidence_mask=mask,
        evaluation_mask=evaluation,
        mode=mode,
    )


def _out_of_fold_view_probabilities(
    train_clean_embeddings: np.ndarray,
    train_surface_embeddings: np.ndarray,
    train_truth: np.ndarray,
    train_mentions: list[Mention],
    document_sources: dict[str, str],
    *,
    class_count: int,
    fold_count: int,
    config: dict[str, object],
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    document_ids = [mention.doc_id for mention in train_mentions]
    assignments = document_folds(
        document_ids,
        document_sources,
        fold_count=fold_count,
        seed=seed,
    )
    folds = np.asarray([assignments[document_id] for document_id in document_ids])
    clean = np.full(
        (len(train_truth), class_count),
        np.nan,
        dtype=np.float32,
    )
    surface = np.full_like(clean, np.nan)
    for fold in range(fold_count):
        train_mask = folds != fold
        held_out = folds == fold
        if not train_mask.any() or not held_out.any():
            raise ValueError(f"invalid diagnostic OOF fold {fold}")
        model = fit_local_classifier(
            train_clean_embeddings[train_mask],
            train_truth[train_mask],
            class_count=class_count,
            config=config,
            seed=seed * 100 + fold,
            device=device,
        )
        clean[held_out] = predict_local_probabilities(
            model,
            train_clean_embeddings[held_out],
            device=device,
        )
        surface[held_out] = predict_local_probabilities(
            model,
            train_surface_embeddings[held_out],
            device=device,
        )
        del model
    if not np.isfinite(clean).all() or not np.isfinite(surface).all():
        raise RuntimeError("diagnostic OOF predictions are incomplete")
    return clean, surface, assignments


def _training_prior(truth: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(truth, minlength=class_count).astype(np.float64) + 1.0
    return (counts / counts.sum()).astype(np.float32)


def _propagate_probabilities(
    probabilities: np.ndarray,
    edge_index: np.ndarray,
    *,
    alpha: float,
    update_mask: np.ndarray | None = None,
    allowed_neighbor_mask: np.ndarray | None = None,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("propagation alpha must be in [0, 1]")
    probabilities = np.asarray(probabilities, dtype=np.float32)
    node_count = len(probabilities)
    if update_mask is None:
        updates = np.ones(node_count, dtype=bool)
    else:
        updates = np.asarray(update_mask, dtype=bool)
    if allowed_neighbor_mask is None:
        allowed = np.ones(node_count, dtype=bool)
    else:
        allowed = np.asarray(allowed_neighbor_mask, dtype=bool)
    sums = np.zeros_like(probabilities)
    counts = np.zeros(node_count, dtype=np.float32)
    if edge_index.size:
        sources = edge_index[0]
        targets = edge_index[1]
        keep = allowed[sources]
        np.add.at(sums, targets[keep], probabilities[sources[keep]])
        np.add.at(counts, targets[keep], 1.0)
    eligible = updates & (counts > 0)
    neighbor_mean = np.zeros_like(probabilities)
    neighbor_mean[eligible] = (
        sums[eligible] / counts[eligible, None]
    )
    result = probabilities.copy()
    result[eligible] = (
        (1.0 - alpha) * probabilities[eligible]
        + alpha * neighbor_mean[eligible]
    )
    result /= result.sum(axis=1, keepdims=True).clip(min=1e-12)
    return result.astype(np.float32)


def _select_propagation_alpha(
    truth: np.ndarray,
    probabilities: np.ndarray,
    edge_index: np.ndarray,
    labels: list[str],
    *,
    grid: Iterable[float],
    evaluation_mask: np.ndarray,
    update_mask: np.ndarray | None,
    allowed_neighbor_mask: np.ndarray | None,
    metric: str,
) -> tuple[float, list[dict[str, float]]]:
    mask = np.asarray(evaluation_mask, dtype=bool)
    rows: list[dict[str, float]] = []
    best_alpha = 0.0
    best_score = -math.inf
    for alpha in grid:
        propagated = _propagate_probabilities(
            probabilities,
            edge_index,
            alpha=float(alpha),
            update_mask=update_mask,
            allowed_neighbor_mask=allowed_neighbor_mask,
        )
        metrics = classification_metrics(
            truth[mask],
            propagated.argmax(axis=1)[mask],
            labels,
            include_empty_classes=metric == "macro_f1",
        )
        score = float(metrics[metric])
        rows.append({"alpha": float(alpha), "score": score})
        if score > best_score + 1e-12:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, rows


def _gold_neighbor_vote(
    fallback_probabilities: np.ndarray,
    truth: np.ndarray,
    edge_index: np.ndarray,
    *,
    class_count: int,
    update_mask: np.ndarray,
    allowed_neighbor_mask: np.ndarray,
) -> np.ndarray:
    result = np.asarray(fallback_probabilities, dtype=np.float32).copy()
    votes = np.zeros((len(truth), class_count), dtype=np.float32)
    if edge_index.size:
        sources = edge_index[0]
        targets = edge_index[1]
        keep = allowed_neighbor_mask[sources]
        np.add.at(votes, (targets[keep], truth[sources[keep]]), 1.0)
    totals = votes.sum(axis=1)
    eligible = np.asarray(update_mask, dtype=bool) & (totals > 0)
    result[eligible] = votes[eligible] / totals[eligible, None]
    return result


def _evaluate_probabilities(
    truth: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    mask: np.ndarray,
) -> dict[str, object]:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        raise ValueError("evaluation mask selects no mentions")
    result = classification_metrics(
        truth[selected],
        probabilities.argmax(axis=1)[selected],
        labels,
        include_empty_classes=False,
    )
    result.update(probability_metrics(truth[selected], probabilities[selected]))
    return result


def _transition_summary(
    truth: np.ndarray,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    selected = np.asarray(mask, dtype=bool)
    baseline = baseline_probabilities.argmax(axis=1)
    candidate = candidate_probabilities.argmax(axis=1)
    baseline_wrong = selected & (baseline != truth)
    baseline_right = selected & (baseline == truth)
    corrected = baseline_wrong & (candidate == truth)
    harmed = baseline_right & (candidate != truth)
    support = int(selected.sum())
    return {
        "baseline_errors": int(baseline_wrong.sum()),
        "corrected": int(corrected.sum()),
        "correction_rate": (
            float(corrected.sum() / baseline_wrong.sum())
            if baseline_wrong.any()
            else 0.0
        ),
        "baseline_correct": int(baseline_right.sum()),
        "harmed": int(harmed.sum()),
        "harm_rate": (
            float(harmed.sum() / baseline_right.sum())
            if baseline_right.any()
            else 0.0
        ),
        "net_accuracy_change": float(
            (corrected.sum() - harmed.sum()) / max(1, support)
        ),
    }


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return float("nan"), float("nan")
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    )


def _macro_f1_from_matrices(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None, :, :]
    true_positive = np.diagonal(values, axis1=1, axis2=2).astype(np.float64)
    predicted = values.sum(axis=1).astype(np.float64)
    support = values.sum(axis=2).astype(np.float64)
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
    result = f1.mean(axis=1)
    return result[0] if squeeze else result


def _bootstrap_sign_probability(distribution: np.ndarray) -> float:
    values = np.asarray(distribution)
    non_positive = (np.count_nonzero(values <= 0.0) + 1) / (
        len(values) + 1
    )
    non_negative = (np.count_nonzero(values >= 0.0) + 1) / (
        len(values) + 1
    )
    return float(min(1.0, 2.0 * min(non_positive, non_negative)))


def _centered_bootstrap_p(
    distribution: np.ndarray,
    observed: float,
) -> float:
    values = np.asarray(distribution, dtype=np.float64)
    centered = values - float(observed)
    return float(
        (
            np.count_nonzero(np.abs(centered) >= abs(float(observed)))
            + 1
        )
        / (len(values) + 1)
    )


def _holm_by_family(rows: list[dict[str, object]]) -> None:
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
    for family_rows in by_family.values():
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
            row["holm_adjusted_p"] = running


def _crossed_bootstrap(
    truth: np.ndarray,
    prediction_store: dict[tuple[int, str, int, str], np.ndarray],
    evaluation_masks: dict[tuple[str, int], np.ndarray],
    document_ids: list[str],
    document_sources: dict[str, str],
    labels: list[str],
    contrasts: list[dict[str, object]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    unique_documents = sorted(set(document_ids))
    document_to_index = {
        document_id: index
        for index, document_id in enumerate(unique_documents)
    }
    document_ids_array = np.asarray(document_ids)
    indices_by_document = {
        document_id: np.flatnonzero(document_ids_array == document_id)
        for document_id in unique_documents
    }
    by_source: dict[str, list[int]] = defaultdict(list)
    for document_id in unique_documents:
        by_source[document_sources[document_id]].append(
            document_to_index[document_id]
        )
    source_groups = [
        np.asarray(indices, dtype=np.int64)
        for _, indices in sorted(by_source.items())
    ]
    optimization_seeds = sorted(
        {key[0] for key in prediction_store}
    )
    class_count = len(labels)

    def cache_for(
        condition: str,
        realization: int,
        model: str,
        optimization_seed: int,
        metric: str,
    ) -> np.ndarray:
        prediction = prediction_store[
            (optimization_seed, condition, realization, model)
        ]
        mask = np.asarray(
            evaluation_masks[(condition, realization)],
            dtype=bool,
        )
        if metric == "macro_f1":
            result = np.zeros(
                (len(unique_documents), class_count, class_count),
                dtype=np.int64,
            )
            for document_id, indices in indices_by_document.items():
                selected = indices[mask[indices]]
                np.add.at(
                    result[document_to_index[document_id]],
                    (truth[selected], prediction[selected]),
                    1,
                )
            return result
        if metric == "accuracy":
            result = np.zeros((len(unique_documents), 2), dtype=np.int64)
            for document_id, indices in indices_by_document.items():
                selected = indices[mask[indices]]
                row = document_to_index[document_id]
                result[row, 0] = int(
                    np.count_nonzero(prediction[selected] == truth[selected])
                )
                result[row, 1] = len(selected)
            return result
        raise ValueError(f"unknown bootstrap metric {metric!r}")

    rows: list[dict[str, object]] = []
    root_generator = np.random.default_rng(seed)
    for contrast_index, contrast in enumerate(contrasts):
        metric = str(contrast["metric"])
        terms = [
            {
                "coefficient": float(term["coefficient"]),
                "condition": str(term["condition"]),
                "model": str(term["model"]),
            }
            for term in contrast["terms"]
        ]
        realizations_by_condition = {
            term["condition"]: sorted(
                realization
                for condition, realization in evaluation_masks
                if condition == term["condition"]
                and all(
                    (
                        optimization_seed,
                        condition,
                        realization,
                        term["model"],
                    )
                    in prediction_store
                    for optimization_seed in optimization_seeds
                )
            )
            for term in terms
        }
        common_realizations = sorted(
            set.intersection(
                *[
                    set(values)
                    for values in realizations_by_condition.values()
                ]
            )
        )
        if not common_realizations:
            continue
        caches = {
            (
                term_index,
                seed_index,
                realization_index,
            ): cache_for(
                term["condition"],
                realization,
                term["model"],
                optimization_seed,
                metric,
            )
            for term_index, term in enumerate(terms)
            for seed_index, optimization_seed in enumerate(optimization_seeds)
            for realization_index, realization in enumerate(common_realizations)
        }
        stack = np.stack(
            [
                np.stack(
                    [
                        np.stack(
                            [
                                caches[
                                    (
                                        term_index,
                                        seed_index,
                                        realization_index,
                                    )
                                ]
                                for realization_index in range(
                                    len(common_realizations)
                                )
                            ],
                            axis=0,
                        )
                        for seed_index in range(len(optimization_seeds))
                    ],
                    axis=0,
                )
                for term_index in range(len(terms))
            ],
            axis=0,
        )

        def metric_values(values: np.ndarray) -> np.ndarray:
            if metric == "macro_f1":
                return _macro_f1_from_matrices(values)
            return np.divide(
                values[:, 0],
                values[:, 1],
                out=np.zeros(values.shape[0], dtype=np.float64),
                where=values[:, 1] > 0,
            )

        observed_seed_values: list[float] = []
        all_documents = np.arange(len(unique_documents), dtype=np.int64)
        for seed_index in range(len(optimization_seeds)):
            realization_values: list[float] = []
            for realization_index in range(len(common_realizations)):
                value = 0.0
                for term_index, term in enumerate(terms):
                    aggregate = stack[
                        term_index,
                        seed_index,
                        realization_index,
                        all_documents,
                    ].sum(axis=0)
                    if metric == "macro_f1":
                        term_value = float(
                            _macro_f1_from_matrices(aggregate)
                        )
                    else:
                        term_value = (
                            float(aggregate[0] / aggregate[1])
                            if aggregate[1]
                            else 0.0
                        )
                    value += float(term["coefficient"]) * term_value
                realization_values.append(value)
            observed_seed_values.append(float(np.mean(realization_values)))

        generator = np.random.default_rng(
            root_generator.integers(0, np.iinfo(np.int64).max)
            + contrast_index
        )
        distribution = np.empty(samples, dtype=np.float64)
        batch_size = 200
        for batch_left in range(0, samples, batch_size):
            batch_right = min(samples, batch_left + batch_size)
            current_size = batch_right - batch_left
            sampled_documents = np.concatenate(
                [
                    generator.choice(
                        group,
                        size=(current_size, len(group)),
                        replace=True,
                    )
                    for group in source_groups
                ],
                axis=1,
            )
            slot_values = np.zeros(
                (current_size, len(optimization_seeds)),
                dtype=np.float64,
            )
            for slot in range(len(optimization_seeds)):
                sampled_seed = generator.integers(
                    0,
                    len(optimization_seeds),
                    size=current_size,
                )
                sampled_realization = generator.integers(
                    0,
                    len(common_realizations),
                    size=current_size,
                )
                for term_index, term in enumerate(terms):
                    selected = stack[
                        term_index,
                        sampled_seed[:, None],
                        sampled_realization[:, None],
                        sampled_documents,
                    ].sum(axis=1)
                    slot_values[:, slot] += (
                        float(term["coefficient"])
                        * metric_values(selected)
                    )
            distribution[batch_left:batch_right] = slot_values.mean(axis=1)
        observed = float(np.mean(observed_seed_values))
        ci95 = np.quantile(distribution, [0.025, 0.975])
        ci90 = np.quantile(distribution, [0.05, 0.95])
        sesoi = float(contrast["sesoi"])
        rows.append(
            {
                "family": contrast["family"],
                "contrast": contrast["contrast"],
                "metric": metric,
                "terms": terms,
                "optimization_seeds": len(optimization_seeds),
                "realizations": len(common_realizations),
                "documents": len(unique_documents),
                "samples": samples,
                "observed_mean_contrast": observed,
                "seed_contrasts": observed_seed_values,
                "bootstrap_mean_contrast": float(distribution.mean()),
                "confidence_interval_95": [
                    float(ci95[0]),
                    float(ci95[1]),
                ],
                "confidence_interval_90": [
                    float(ci90[0]),
                    float(ci90[1]),
                ],
                "two_sided_bootstrap_p": _centered_bootstrap_p(
                    distribution,
                    observed,
                ),
                "bootstrap_sign_probability_two_sided": (
                    _bootstrap_sign_probability(distribution)
                ),
                "sesoi": sesoi,
                "ci90_inside_sesoi_bounds_descriptive": bool(
                    ci90[0] > -sesoi and ci90[1] < sesoi
                ),
                "resampling": (
                    "optimization seeds crossed with source-stratified "
                    "documents; corruption realizations nested within seed"
                ),
                **(
                    {
                        "noninferiority_margin": float(
                            contrast["noninferiority_margin"]
                        ),
                        "noninferior_by_95ci": bool(
                            ci95[0]
                            > -float(contrast["noninferiority_margin"])
                        ),
                    }
                    if "noninferiority_margin" in contrast
                    else {}
                ),
            }
        )
    _holm_by_family(rows)
    for row in rows:
        ci95 = row["confidence_interval_95"]
        observed = float(row["observed_mean_contrast"])
        sesoi = float(row["sesoi"])
        adjusted = float(row["holm_adjusted_p"])
        if "noninferiority_margin" in row:
            decision = (
                "noninferiority_supported_by_conservative_95ci_descriptive"
                if bool(row["noninferior_by_95ci"])
                else "noninferiority_inconclusive"
            )
        elif (
            adjusted < 0.05
            and float(ci95[0]) > 0.0
            and observed >= sesoi
        ):
            decision = "supported_benefit"
        elif float(ci95[0]) >= sesoi:
            decision = "material_benefit"
        elif bool(row["ci90_inside_sesoi_bounds_descriptive"]):
            decision = "inconclusive_inside_sesoi_descriptive_not_tost"
        elif float(ci95[1]) <= -sesoi:
            decision = "material_harm"
        else:
            decision = "inconclusive"
        row["decision"] = decision
    return rows


SUMMARY_METRICS = (
    "accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "brier_score",
    "ece_10_bins",
    "correction_rate",
    "harm_rate",
    "net_accuracy_change",
)


def _summarize_records(
    records: list[dict[str, object]],
    *,
    keys: tuple[str, ...],
    metrics: tuple[str, ...] = SUMMARY_METRICS,
) -> list[dict[str, object]]:
    by_seed: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_seed[
            tuple(record[key] for key in keys) + (record["seed"],)
        ].append(record)
    seed_rows: list[dict[str, object]] = []
    for grouped_key, grouped_records in sorted(
        by_seed.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        row: dict[str, object] = {
            key: value
            for key, value in zip(keys, grouped_key[:-1], strict=True)
        }
        row["seed"] = grouped_key[-1]
        row["realizations"] = len(grouped_records)
        for metric in metrics:
            available = [
                float(record[metric])
                for record in grouped_records
                if metric in record
            ]
            if available:
                row[metric] = float(np.mean(available))
        for metadata in (
            "support",
            "documents",
            "deployable",
            "oracle_only",
            "uses_test_gold",
            "train_fraction",
            "checkpoint_condition",
            "primary_metric",
        ):
            if metadata in grouped_records[0]:
                values = [record.get(metadata) for record in grouped_records]
                if metadata in {"support", "documents"}:
                    row[metadata] = float(
                        np.mean([float(value) for value in values])
                    )
                else:
                    row[metadata] = values[0]
        seed_rows.append(row)
    final_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        final_groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, object]] = []
    for grouped_key, grouped_rows in sorted(
        final_groups.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        row = {
            key: value
            for key, value in zip(keys, grouped_key, strict=True)
        }
        row["optimization_seeds"] = len(grouped_rows)
        row["realizations_per_seed"] = grouped_rows[0]["realizations"]
        for metric in metrics:
            values = [
                float(grouped[metric])
                for grouped in grouped_rows
                if metric in grouped
            ]
            if values:
                mean, standard_deviation = _mean_std(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = standard_deviation
        for metadata in (
            "support",
            "documents",
            "deployable",
            "oracle_only",
            "uses_test_gold",
            "train_fraction",
            "checkpoint_condition",
            "primary_metric",
        ):
            if metadata in grouped_rows[0]:
                row[metadata] = grouped_rows[0][metadata]
        output.append(row)
    return output


def _surface_ambiguity_mask(
    train_mentions: list[Mention],
    target_mentions: list[Mention],
    *,
    minimum_support: int,
    minimum_entropy_bits: float,
) -> tuple[np.ndarray, dict[str, dict[str, float | int]]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for mention in train_mentions:
        counts[normalize_surface(mention.text)][mention.label] += 1
    statistics: dict[str, dict[str, float | int]] = {}
    ambiguous: set[str] = set()
    for surface, label_counts in counts.items():
        support = sum(label_counts.values())
        probabilities = np.asarray(
            list(label_counts.values()),
            dtype=np.float64,
        )
        probabilities /= probabilities.sum()
        entropy = float(
            -(probabilities * np.log2(probabilities.clip(min=1e-12))).sum()
        )
        majority_rate = float(max(label_counts.values()) / support)
        statistics[surface] = {
            "support": support,
            "labels": len(label_counts),
            "entropy_bits": entropy,
            "majority_rate": majority_rate,
        }
        if (
            support >= minimum_support
            and entropy >= minimum_entropy_bits
        ):
            ambiguous.add(surface)
    mask = np.asarray(
        [
            normalize_surface(mention.text) in ambiguous
            for mention in target_mentions
        ],
        dtype=bool,
    )
    return mask, statistics


def _predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    return -(
        values * np.log(values.clip(min=1e-12))
    ).sum(axis=1)


def _ambiguity_cohorts(
    train_mentions: list[Mention],
    test_mentions: list[Mention],
    validation_n0: np.ndarray,
    test_n0: np.ndarray,
    *,
    minimum_surface_support: int,
    minimum_surface_entropy_bits: float,
    uncertainty_quantile: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    target, repeated, repeat_counts = _repeat_masks(
        test_mentions,
        target_seed=0,
    )
    del target
    train_ambiguous, surface_statistics = _surface_ambiguity_mask(
        train_mentions,
        test_mentions,
        minimum_support=minimum_surface_support,
        minimum_entropy_bits=minimum_surface_entropy_bits,
    )
    validation_entropy = _predictive_entropy(validation_n0)
    threshold = float(
        np.quantile(validation_entropy, uncertainty_quantile)
    )
    uncertain = _predictive_entropy(test_n0) >= threshold
    seen_surfaces = set(surface_statistics)
    seen = np.asarray(
        [
            normalize_surface(mention.text) in seen_surfaces
            for mention in test_mentions
        ],
        dtype=bool,
    )
    cohorts = {
        "all": np.ones(len(test_mentions), dtype=bool),
        "repeated": repeated,
        "singleton": ~repeated,
        "train_ambiguous": train_ambiguous,
        "train_ambiguous_repeated": train_ambiguous & repeated,
        "uncertain": uncertain,
        "uncertain_repeated": uncertain & repeated,
        "uncertain_singleton": uncertain & ~repeated,
        "seen_surface": seen,
        "unseen_surface": ~seen,
    }
    metadata = {
        "validation_uncertainty_quantile": uncertainty_quantile,
        "validation_entropy_threshold_nats": threshold,
        "train_ambiguous_minimum_support": minimum_surface_support,
        "train_ambiguous_minimum_entropy_bits": (
            minimum_surface_entropy_bits
        ),
        "train_surface_statistics_count": len(surface_statistics),
        "repeat_counts": repeat_counts,
    }
    return cohorts, metadata


def _edge_count(edge_indices: dict[str, np.ndarray]) -> int:
    undirected = _undirected_edge_set(edge_indices)
    return len(undirected)


def _undirected_edge_set(
    edge_indices: dict[str, np.ndarray],
) -> set[tuple[int, int]]:
    undirected: set[tuple[int, int]] = set()
    for edges in edge_indices.values():
        for left, right in edges.T:
            left_value = int(left)
            right_value = int(right)
            if left_value != right_value:
                undirected.add(
                    (
                        min(left_value, right_value),
                        max(left_value, right_value),
                    )
                )
    return undirected


def _degree_vector(
    edge_indices: dict[str, np.ndarray],
    node_count: int,
) -> np.ndarray:
    degree = np.zeros(node_count, dtype=np.int64)
    for left, right in _undirected_edge_set(edge_indices):
        degree[left] += 1
        degree[right] += 1
    return degree


def _edge_homophily(
    edge_indices: dict[str, np.ndarray],
    truth: np.ndarray,
) -> float:
    pairs: set[tuple[int, int]] = set()
    for edges in edge_indices.values():
        for left, right in edges.T:
            left_value = int(left)
            right_value = int(right)
            if left_value < right_value:
                pairs.add((left_value, right_value))
    if not pairs:
        return float("nan")
    return float(
        np.mean([truth[left] == truth[right] for left, right in pairs])
    )


def _prepare_local_views(
    train_documents: list[Document],
    clean_lookup: dict[str, np.ndarray],
    surface_lookup: dict[str, np.ndarray],
    validation_clean_embeddings: np.ndarray,
    validation_surface_embeddings: np.ndarray,
    test_clean_embeddings: np.ndarray,
    test_surface_embeddings: np.ndarray,
    validation_truth: np.ndarray,
    test_truth: np.ndarray,
    validation_mentions: list[Mention],
    test_mentions: list[Mention],
    label_to_id: dict[str, int],
    document_sources: dict[str, str],
    *,
    near_threshold: int,
    alias_threshold: float,
    class_count: int,
    fold_count: int,
    local_config: dict[str, object],
    seed: int,
    device: str,
) -> LocalViews:
    train_mentions, train_clean_embeddings = _split_embeddings(
        train_documents,
        clean_lookup,
    )
    surface_mentions, train_surface_embeddings = _split_embeddings(
        train_documents,
        surface_lookup,
    )
    if [mention.mention_id for mention in train_mentions] != [
        mention.mention_id for mention in surface_mentions
    ]:
        raise RuntimeError("clean and surface train embeddings are misaligned")
    train_truth = _labels_array(train_mentions, label_to_id)
    train_source = build_relational_source(
        train_documents,
        label_to_id=label_to_id,
        near_threshold_tokens=near_threshold,
        alias_threshold=alias_threshold,
    )
    if train_source.mention_ids != [
        mention.mention_id for mention in train_mentions
    ]:
        raise RuntimeError("diagnostic train graph is misaligned")
    oof_clean, oof_surface, assignments = (
        _out_of_fold_view_probabilities(
            train_clean_embeddings,
            train_surface_embeddings,
            train_truth,
            train_mentions,
            document_sources,
            class_count=class_count,
            fold_count=fold_count,
            config=local_config,
            seed=seed,
            device=device,
        )
    )
    local_model = fit_local_classifier(
        train_clean_embeddings,
        train_truth,
        class_count=class_count,
        config=local_config,
        seed=seed,
        device=device,
    )
    validation_clean = predict_local_probabilities(
        local_model,
        validation_clean_embeddings,
        device=device,
    )
    validation_surface = predict_local_probabilities(
        local_model,
        validation_surface_embeddings,
        device=device,
    )
    test_clean = predict_local_probabilities(
        local_model,
        test_clean_embeddings,
        device=device,
    )
    test_surface = predict_local_probabilities(
        local_model,
        test_surface_embeddings,
        device=device,
    )
    if len(validation_clean) != len(validation_truth) or len(test_clean) != len(
        test_truth
    ):
        raise RuntimeError("diagnostic local predictions are misaligned")
    if [mention.mention_id for mention in validation_mentions] != list(
        dict.fromkeys(mention.mention_id for mention in validation_mentions)
    ):
        raise RuntimeError("validation mention IDs must be unique")
    if [mention.mention_id for mention in test_mentions] != list(
        dict.fromkeys(mention.mention_id for mention in test_mentions)
    ):
        raise RuntimeError("test mention IDs must be unique")
    return LocalViews(
        train_mentions=train_mentions,
        train_source=train_source,
        train_truth=train_truth,
        train_clean_embeddings=train_clean_embeddings,
        train_surface_embeddings=train_surface_embeddings,
        train_clean_probabilities=oof_clean,
        train_surface_probabilities=oof_surface,
        validation_clean_probabilities=validation_clean,
        validation_surface_probabilities=validation_surface,
        test_clean_probabilities=test_clean,
        test_surface_probabilities=test_surface,
        train_prior=_training_prior(train_truth, class_count),
        local_model=local_model,
        fold_assignments=assignments,
    )


def _edge_regimes(
    source: RelationalGraphSource,
    *,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    typed = materialize_relational_batch(
        source,
        configuration="typed_all",
        seed=seed,
    )
    repeat = as_single_relation_edge_indices(
        typed.edge_indices["repeat"],
        relation=SINGLE_RELATION,
    )
    semantic_union_index = _union_edge_index(
        typed.edge_indices,
        ("sent", "repeat", "near"),
    )
    semantic_union = as_single_relation_edge_indices(
        semantic_union_index,
        relation=SINGLE_RELATION,
    )
    pruned = semantic_union_pruned_to_same_label_edge_indices(
        typed.edge_indices,
        source.labels,
        source.document_ids,
        relations=("sent", "repeat", "near"),
        relation=SINGLE_RELATION,
    )
    oracle = build_sparse_gold_oracle_edge_indices(
        source.labels,
        source.document_ids,
        relation=SINGLE_RELATION,
    )
    return {
        "N0_node_only": empty_edge_indices(),
        "G_repeat": repeat,
        "G_semantic_union": semantic_union,
        "O_pruned_union": pruned,
        "O_label_sparse": oracle,
    }


MODEL_METADATA: dict[str, dict[str, object]] = {
    "local": {
        "deployable": True,
        "oracle_only": False,
        "uses_test_gold": False,
    },
    "N0_node_only": {
        "deployable": True,
        "oracle_only": False,
        "uses_test_gold": False,
    },
    "G_repeat": {
        "deployable": True,
        "oracle_only": False,
        "uses_test_gold": False,
    },
    "G_lemma": {
        "deployable": True,
        "oracle_only": False,
        "uses_test_gold": False,
    },
    "G_semantic_union": {
        "deployable": True,
        "oracle_only": False,
        "uses_test_gold": False,
    },
    "O_pruned_union": {
        "deployable": False,
        "oracle_only": True,
        "uses_test_gold": True,
    },
    "O_label_sparse": {
        "deployable": False,
        "oracle_only": True,
        "uses_test_gold": True,
    },
    "AVG_repeat": {
        "deployable": True,
        "oracle_only": False,
        "uses_test_gold": False,
    },
    "AVG_lemma": {
        "deployable": True,
        "oracle_only": False,
        "uses_test_gold": False,
    },
    "AVG_label_sparse": {
        "deployable": False,
        "oracle_only": True,
        "uses_test_gold": True,
    },
    "GoldVote_repeat": {
        "deployable": False,
        "oracle_only": True,
        "uses_test_gold": True,
    },
    "GoldVote_label_sparse": {
        "deployable": False,
        "oracle_only": True,
        "uses_test_gold": True,
    },
    "G_repeat_no_edges_cf": {
        "deployable": False,
        "oracle_only": False,
        "uses_test_gold": False,
    },
    "O_label_sparse_no_edges_cf": {
        "deployable": False,
        "oracle_only": True,
        "uses_test_gold": True,
    },
}


def _record_evaluation(
    records: list[dict[str, object]],
    prediction_store: dict[tuple[int, str, int, str], np.ndarray],
    truth: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
    labels: list[str],
    mask: np.ndarray,
    *,
    seed: int,
    training_scenario: str,
    condition: str,
    realization: int,
    train_fraction: float,
    checkpoint_condition: str,
    primary_metric: str,
    document_ids: list[str],
) -> None:
    baseline = probabilities_by_model["N0_node_only"]
    selected_documents = len(
        {
            document_id
            for document_id, selected in zip(
                document_ids,
                np.asarray(mask, dtype=bool),
                strict=True,
            )
            if selected
        }
    )
    for model, probabilities in probabilities_by_model.items():
        metrics = _evaluate_probabilities(truth, probabilities, labels, mask)
        metadata = MODEL_METADATA.get(
            model,
            {
                "deployable": False,
                "oracle_only": False,
                "uses_test_gold": False,
            },
        )
        transition = (
            _transition_summary(truth, baseline, probabilities, mask)
            if model != "N0_node_only"
            else {
                "baseline_errors": int(
                    np.count_nonzero(
                        np.asarray(mask, dtype=bool)
                        & (baseline.argmax(axis=1) != truth)
                    )
                ),
                "corrected": 0,
                "correction_rate": 0.0,
                "baseline_correct": int(
                    np.count_nonzero(
                        np.asarray(mask, dtype=bool)
                        & (baseline.argmax(axis=1) == truth)
                    )
                ),
                "harmed": 0,
                "harm_rate": 0.0,
                "net_accuracy_change": 0.0,
            }
        )
        records.append(
            {
                "seed": seed,
                "training_scenario": training_scenario,
                "condition": condition,
                "realization": realization,
                "train_fraction": train_fraction,
                "checkpoint_condition": checkpoint_condition,
                "model": model,
                "primary_metric": primary_metric,
                "support": int(np.asarray(mask, dtype=bool).sum()),
                "documents": selected_documents,
                **metadata,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "negative_log_likelihood": metrics[
                    "negative_log_likelihood"
                ],
                "brier_score": metrics["brier_score"],
                "ece_10_bins": metrics["ece_10_bins"],
                **transition,
            }
        )
        prediction_store[(seed, condition, realization, model)] = (
            probabilities.argmax(axis=1).astype(np.int16)
        )


def run_diagnostic_experiment(
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
    diagnostic = config["diagnostics"]
    near_threshold = int(config["graph"].get("near_threshold_tokens", 50))
    alias_threshold = float(diagnostic.get("alias_threshold", 0.9))
    seeds = [
        int(value)
        for value in diagnostic.get("optimization_seeds", DIAGNOSTIC_SEEDS)
    ]
    corruption_seeds = [
        int(value)
        for value in diagnostic.get(
            "corruption_seeds",
            DEFAULT_CORRUPTION_SEEDS,
        )
    ]
    fractions = sorted(
        {
            float(value)
            for value in diagnostic.get(
                "low_resource_fractions",
                DEFAULT_RESOURCE_FRACTIONS,
            )
        }
    )
    if 1.0 not in fractions:
        raise ValueError("diagnostics require a 1.0 supervision fraction")
    model_config = diagnostic["model"]
    local_config = config["training"]["local"]
    fold_count = int(config["training"].get("oof_folds", 5))
    alpha_grid = [
        float(value)
        for value in diagnostic.get(
            "propagation_alpha_grid",
            PROPAGATION_ALPHA_GRID,
        )
    ]
    bootstrap_samples = int(
        diagnostic.get(
            "bootstrap_samples",
            config.get("statistics", {}).get("bootstrap_samples", 20000),
        )
    )
    bootstrap_seed = int(
        diagnostic.get(
            "bootstrap_seed",
            config.get("statistics", {}).get("bootstrap_seed", 8675309)
            + 400000,
        )
    )
    subset_seed = int(diagnostic.get("low_resource_subset_seed", 271828))
    train_target_seed = int(diagnostic.get("train_target_seed", 314159))
    validation_target_seed = int(
        diagnostic.get("validation_target_seed", 161803)
    )
    topology_seeds = [
        int(value)
        for value in diagnostic.get(
            "topology_seeds",
            DEFAULT_CORRUPTION_SEEDS,
        )
    ]

    cache_root = project_root / str(
        config.get("embedding_cache", "data/cache/embeddings")
    )
    model_cache = project_root / str(config.get("model_cache", "data/models"))
    clean_ids, clean_embeddings, clean_cache_path = encode_with_cache(
        documents,
        encoder=config["encoder"],
        cache_root=cache_root,
        model_cache=model_cache,
        device=device,
    )
    surface_ids, surface_embeddings, surface_cache_path = encode_with_cache(
        documents,
        encoder=config["encoder"],
        cache_root=cache_root,
        model_cache=model_cache,
        device=device,
        noise={
            "kind": "context_dropout",
            "level": 1.0,
            "seed": 0,
        },
    )
    if clean_ids != surface_ids:
        raise RuntimeError("clean and surface-only caches are misaligned")
    clean_lookup = _embedding_lookup(clean_ids, clean_embeddings)
    surface_lookup = _embedding_lookup(surface_ids, surface_embeddings)
    validation_mentions, validation_clean_embeddings = _split_embeddings(
        validation_documents,
        clean_lookup,
    )
    _, validation_surface_embeddings = _split_embeddings(
        validation_documents,
        surface_lookup,
    )
    test_mentions, test_clean_embeddings = _split_embeddings(
        test_documents,
        clean_lookup,
    )
    _, test_surface_embeddings = _split_embeddings(
        test_documents,
        surface_lookup,
    )
    validation_truth = _labels_array(validation_mentions, label_to_id)
    test_truth = _labels_array(test_mentions, label_to_id)
    validation_source = build_relational_source(
        validation_documents,
        label_to_id=label_to_id,
        near_threshold_tokens=near_threshold,
        alias_threshold=alias_threshold,
    )
    test_source = build_relational_source(
        test_documents,
        label_to_id=label_to_id,
        near_threshold_tokens=near_threshold,
        alias_threshold=alias_threshold,
    )
    if validation_source.mention_ids != [
        mention.mention_id for mention in validation_mentions
    ]:
        raise RuntimeError("validation graph is misaligned")
    if test_source.mention_ids != [
        mention.mention_id for mention in test_mentions
    ]:
        raise RuntimeError("test graph is misaligned")
    document_sources = {
        document.doc_id: document.source for document in documents
    }
    test_document_ids = [mention.doc_id for mention in test_mentions]

    subset_ids = stable_nested_document_subsets(
        documents,
        fractions,
        seed=subset_seed,
    )
    subset_documents = {
        fraction: [
            document
            for document in train_documents
            if document.doc_id in set(subset_ids[fraction])
        ]
        for fraction in fractions
    }
    subset_diagnostics: list[dict[str, object]] = []
    for fraction in fractions:
        selected_mentions = flatten_mentions(subset_documents[fraction])
        counts = Counter(mention.label for mention in selected_mentions)
        subset_diagnostics.append(
            {
                "fraction": fraction,
                "subset_seed": subset_seed,
                "documents": len(subset_documents[fraction]),
                "mentions": len(selected_mentions),
                "classes_present": len(counts),
                "missing_classes": sorted(set(labels) - set(counts)),
                "source_counts": dict(
                    sorted(
                        Counter(
                            document.source
                            for document in subset_documents[fraction]
                        ).items()
                    )
                ),
                "label_counts": dict(sorted(counts.items())),
                "document_ids": list(subset_ids[fraction]),
            }
        )
    save_json(
        output_directory / "low_resource_subsets.json",
        subset_diagnostics,
    )
    write_csv(
        output_directory / "table_g0_low_resource_subsets.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"document_ids", "label_counts"}
            }
            for row in subset_diagnostics
        ],
    )

    target_selection: dict[str, object] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    validation_target, validation_component, validation_target_counts = (
        _repeat_masks(
            validation_mentions,
            target_seed=validation_target_seed,
        )
    )
    target_selection["validation"] = {
        "seed": validation_target_seed,
        **validation_target_counts,
    }
    test_masks: dict[
        int,
        tuple[np.ndarray, np.ndarray, dict[str, int]],
    ] = {}
    for corruption_seed in corruption_seeds:
        masks = _repeat_masks(
            test_mentions,
            target_seed=corruption_seed,
        )
        test_masks[corruption_seed] = masks
        target_selection["test"][str(corruption_seed)] = {
            "seed": corruption_seed,
            **masks[2],
        }

    metric_records: list[dict[str, object]] = []
    alpha_records: list[dict[str, object]] = []
    counterfactual_records: list[dict[str, object]] = []
    graph_records: list[dict[str, object]] = []
    ambiguity_records: list[dict[str, object]] = []
    ambiguity_metadata: list[dict[str, object]] = []
    clean_test_probabilities_by_seed: dict[
        int,
        dict[str, np.ndarray],
    ] = {}
    clean_validation_n0_by_seed: dict[int, np.ndarray] = {}
    prediction_store: dict[
        tuple[int, str, int, str],
        np.ndarray,
    ] = {}
    evaluation_masks: dict[tuple[str, int], np.ndarray] = {}
    for fraction in fractions:
        evaluation_masks[(f"supervision_{fraction:.2f}", 0)] = np.ones(
            len(test_truth),
            dtype=bool,
        )
    for corruption_seed, (target, _, _) in test_masks.items():
        for condition in (
            "clean_targets",
            "target_surface",
            "component_surface",
            "target_prior",
            "component_prior",
        ):
            evaluation_masks[(condition, corruption_seed)] = target.copy()

    test_edge_regimes = _edge_regimes(test_source, seed=0)
    validation_edge_regimes = _edge_regimes(validation_source, seed=0)
    randomized_test_edges = {
        model_name: {
            topology_seed: (
                degree_preserving_randomize_edge_indices_by_document(
                    test_edge_regimes[model_name],
                    test_source.document_ids,
                    seed=topology_seed,
                    relation=SINGLE_RELATION,
                )
            )
            for topology_seed in topology_seeds
        }
        for model_name in ("G_repeat", "O_label_sparse")
    }
    control_strength_rows: list[dict[str, object]] = []
    for model_name, controls in randomized_test_edges.items():
        original = _undirected_edge_set(test_edge_regimes[model_name])
        original_degrees = _degree_vector(
            test_edge_regimes[model_name],
            len(test_truth),
        )
        for topology_seed, controlled_edges in controls.items():
            controlled = _undirected_edge_set(controlled_edges)
            retained = len(original & controlled)
            control_strength_rows.append(
                {
                    "model": model_name,
                    "topology_seed": topology_seed,
                    "edges": len(original),
                    "retained_edges": retained,
                    "retained_fraction": (
                        retained / len(original) if original else 0.0
                    ),
                    "changed_fraction": (
                        1.0 - retained / len(original)
                        if original
                        else 0.0
                    ),
                    "per_node_degree_preserved": bool(
                        np.array_equal(
                            original_degrees,
                            _degree_vector(
                                controlled_edges,
                                len(test_truth),
                            ),
                        )
                    ),
                    "original_label_homophily": _edge_homophily(
                        test_edge_regimes[model_name],
                        test_truth,
                    ),
                    "controlled_label_homophily": _edge_homophily(
                        controlled_edges,
                        test_truth,
                    ),
                }
            )
    write_csv(
        output_directory / "table_g8_control_strength.csv",
        control_strength_rows,
    )
    graph_regime_metadata = {
        "N0_node_only": {
            "deployable": True,
            "oracle_only": False,
        },
        "G_repeat": {
            "deployable": True,
            "oracle_only": False,
        },
        "G_semantic_union": {
            "deployable": True,
            "oracle_only": False,
        },
        "O_pruned_union": {
            "deployable": False,
            "oracle_only": True,
        },
        "O_label_sparse": {
            "deployable": False,
            "oracle_only": True,
        },
    }

    for seed in seeds:
        seed_directory = output_directory / f"seed-{seed}"
        seed_directory.mkdir(parents=True, exist_ok=True)
        local_by_fraction: dict[float, LocalViews] = {}
        for fraction in fractions:
            local_by_fraction[fraction] = _prepare_local_views(
                subset_documents[fraction],
                clean_lookup,
                surface_lookup,
                validation_clean_embeddings,
                validation_surface_embeddings,
                test_clean_embeddings,
                test_surface_embeddings,
                validation_truth,
                test_truth,
                validation_mentions,
                test_mentions,
                label_to_id,
                document_sources,
                near_threshold=near_threshold,
                alias_threshold=alias_threshold,
                class_count=len(labels),
                fold_count=fold_count,
                local_config=local_config,
                seed=seed,
                device=device,
            )

        scenario_specs: list[dict[str, object]] = [
            {
                "training_scenario": f"supervision_{fraction:.2f}",
                "fraction": fraction,
                "mode": "identity",
                "validation_metric": "macro_f1",
            }
            for fraction in fractions
        ] + [
            {
                "training_scenario": "target_surface",
                "fraction": 1.0,
                "mode": "surface",
                "validation_metric": "accuracy",
            },
            {
                "training_scenario": "target_prior",
                "fraction": 1.0,
                "mode": "prior",
                "validation_metric": "accuracy",
            },
        ]
        clean_full_probabilities: dict[str, np.ndarray] | None = None
        clean_full_validation: dict[str, np.ndarray] | None = None

        for scenario in scenario_specs:
            scenario_name = str(scenario["training_scenario"])
            fraction = float(scenario["fraction"])
            mode = str(scenario["mode"])
            validation_metric = str(scenario["validation_metric"])
            local = local_by_fraction[fraction]
            train_target, train_component, train_target_counts = _repeat_masks(
                local.train_mentions,
                target_seed=train_target_seed,
            )
            if fraction == 1.0:
                target_selection["train"][str(seed)] = {
                    "seed": train_target_seed,
                    **train_target_counts,
                }
            if mode == "identity":
                train_evidence_mask = np.zeros(len(local.train_truth), dtype=bool)
                validation_evidence_mask = np.zeros(
                    len(validation_truth),
                    dtype=bool,
                )
                train_evaluation_mask = np.ones(
                    len(local.train_truth),
                    dtype=bool,
                )
                validation_evaluation_mask = np.ones(
                    len(validation_truth),
                    dtype=bool,
                )
            else:
                train_evidence_mask = train_target
                validation_evidence_mask = validation_target
                train_evaluation_mask = train_target
                validation_evaluation_mask = validation_target
            train_inputs = _evidence_inputs(
                local.train_clean_embeddings,
                local.train_surface_embeddings,
                local.train_clean_probabilities,
                local.train_surface_probabilities,
                local.train_source,
                mode=mode,
                evidence_mask=train_evidence_mask,
                evaluation_mask=train_evaluation_mask,
                train_prior=local.train_prior,
            )
            validation_inputs = _evidence_inputs(
                validation_clean_embeddings,
                validation_surface_embeddings,
                local.validation_clean_probabilities,
                local.validation_surface_probabilities,
                validation_source,
                mode=mode,
                evidence_mask=validation_evidence_mask,
                evaluation_mask=validation_evaluation_mask,
                train_prior=local.train_prior,
            )
            train_edges = _edge_regimes(local.train_source, seed=seed)
            arms = [
                "N0_node_only",
                "G_repeat",
                "O_label_sparse",
            ]
            if scenario_name == "supervision_1.00":
                arms.extend(["G_semantic_union", "O_pruned_union"])
            results: dict[str, object] = {}
            validation_probabilities: dict[str, np.ndarray] = {
                "local": validation_inputs.local_probabilities
            }
            for model_name in arms:
                result = fit_relational_classifier(
                    train_inputs.features,
                    local.train_truth,
                    train_inputs.local_probabilities,
                    train_edges[model_name],
                    validation_inputs.features,
                    validation_truth,
                    validation_inputs.local_probabilities,
                    validation_edge_regimes[model_name],
                    labels=labels,
                    relation_names=list(RELATION_NAMES),
                    config=model_config,
                    seed=seed,
                    device=device,
                    validation_mask=validation_evaluation_mask,
                    validation_metric=validation_metric,
                )
                results[model_name] = result
                validation_probabilities[model_name] = (
                    predict_relational_probabilities(
                        result,
                        validation_inputs.features,
                        validation_inputs.local_probabilities,
                        validation_edge_regimes[model_name],
                        list(RELATION_NAMES),
                        device=device,
                    )
                )
                graph_records.append(
                    {
                        "seed": seed,
                        "training_scenario": scenario_name,
                        "train_fraction": fraction,
                        "model": model_name,
                        **graph_regime_metadata[model_name],
                        "parameter_count": result.parameter_count,
                        "active_relation_branches": int(
                            model_name != "N0_node_only"
                        ),
                        "best_epoch": result.best_epoch,
                        "validation_metric": result.validation_metric,
                        "best_validation_score": (
                            result.best_validation_score
                        ),
                        "best_validation_macro_f1": (
                            result.best_validation_macro_f1
                        ),
                        "train_edges": _edge_count(
                            train_edges[model_name]
                        ),
                        "validation_edges": _edge_count(
                            validation_edge_regimes[model_name]
                        ),
                        "test_edges": _edge_count(
                            test_edge_regimes[model_name]
                        ),
                        "train_edge_homophily": _edge_homophily(
                            train_edges[model_name],
                            local.train_truth,
                        ),
                        "validation_edge_homophily": _edge_homophily(
                            validation_edge_regimes[model_name],
                            validation_truth,
                        ),
                        "test_edge_homophily": _edge_homophily(
                            test_edge_regimes[model_name],
                            test_truth,
                        ),
                    }
                )
                checkpoint_directory = seed_directory / scenario_name
                checkpoint_directory.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu().clone()
                            for key, value in result.model.state_dict().items()
                        },
                        "labels": labels,
                        "relations": list(RELATION_NAMES),
                        "model_config": model_config,
                        "model": model_name,
                        "training_scenario": scenario_name,
                        "seed": seed,
                        "config_checksum": checksum,
                        "source_checksum": code_checksum,
                        "best_epoch": result.best_epoch,
                        "validation_metric": result.validation_metric,
                        "best_validation_score": (
                            result.best_validation_score
                        ),
                    },
                    checkpoint_directory / f"{model_name.lower()}_model.pt",
                )

            propagation_alphas: dict[str, float] = {}
            for edge_model, output_name in (
                ("G_repeat", "AVG_repeat"),
                ("O_label_sparse", "AVG_label_sparse"),
            ):
                update_mask = (
                    validation_evaluation_mask
                    if validation_metric == "accuracy"
                    else None
                )
                allowed = (
                    ~validation_inputs.evidence_mask
                    if validation_metric == "accuracy"
                    else None
                )
                alpha, alpha_rows = _select_propagation_alpha(
                    validation_truth,
                    validation_probabilities["N0_node_only"],
                    validation_edge_regimes[edge_model][SINGLE_RELATION],
                    labels,
                    grid=alpha_grid,
                    evaluation_mask=validation_evaluation_mask,
                    update_mask=update_mask,
                    allowed_neighbor_mask=allowed,
                    metric=validation_metric,
                )
                propagation_alphas[output_name] = alpha
                alpha_records.extend(
                    {
                        "seed": seed,
                        "training_scenario": scenario_name,
                        "edge_model": edge_model,
                        "propagation_model": output_name,
                        "validation_metric": validation_metric,
                        "selected": abs(row["alpha"] - alpha) < 1e-12,
                        **row,
                    }
                    for row in alpha_rows
                )

            def evaluate_condition(
                *,
                condition: str,
                realization: int,
                evidence_mode: str,
                evidence_mask: np.ndarray,
                evaluation_mask: np.ndarray,
                checkpoint_condition: str,
            ) -> dict[str, np.ndarray]:
                inputs = _evidence_inputs(
                    test_clean_embeddings,
                    test_surface_embeddings,
                    local.test_clean_probabilities,
                    local.test_surface_probabilities,
                    test_source,
                    mode=evidence_mode,
                    evidence_mask=evidence_mask,
                    evaluation_mask=evaluation_mask,
                    train_prior=local.train_prior,
                )
                probabilities: dict[str, np.ndarray] = {
                    "local": inputs.local_probabilities,
                }
                for model_name, result in results.items():
                    probabilities[model_name] = (
                        predict_relational_probabilities(
                            result,
                            inputs.features,
                            inputs.local_probabilities,
                            test_edge_regimes[model_name],
                            list(RELATION_NAMES),
                            device=device,
                        )
                    )
                update_mask = (
                    evaluation_mask
                    if checkpoint_condition != "clean_global"
                    else None
                )
                allowed_neighbors = (
                    ~inputs.evidence_mask
                    if checkpoint_condition != "clean_global"
                    else None
                )
                for edge_model, output_name in (
                    ("G_repeat", "AVG_repeat"),
                    ("O_label_sparse", "AVG_label_sparse"),
                ):
                    probabilities[output_name] = _propagate_probabilities(
                        probabilities["N0_node_only"],
                        test_edge_regimes[edge_model][SINGLE_RELATION],
                        alpha=propagation_alphas[output_name],
                        update_mask=update_mask,
                        allowed_neighbor_mask=allowed_neighbors,
                    )
                gold_update = (
                    evaluation_mask
                    if checkpoint_condition != "clean_global"
                    else np.ones(len(test_truth), dtype=bool)
                )
                gold_allowed = (
                    ~inputs.evidence_mask
                    if checkpoint_condition != "clean_global"
                    else np.ones(len(test_truth), dtype=bool)
                )
                probabilities["GoldVote_repeat"] = _gold_neighbor_vote(
                    probabilities["N0_node_only"],
                    test_truth,
                    test_edge_regimes["G_repeat"][SINGLE_RELATION],
                    class_count=len(labels),
                    update_mask=gold_update,
                    allowed_neighbor_mask=gold_allowed,
                )
                probabilities["GoldVote_label_sparse"] = (
                    _gold_neighbor_vote(
                        probabilities["N0_node_only"],
                        test_truth,
                        test_edge_regimes["O_label_sparse"][
                            SINGLE_RELATION
                        ],
                        class_count=len(labels),
                        update_mask=gold_update,
                        allowed_neighbor_mask=gold_allowed,
                    )
                )
                for full_model, counterfactual_name in (
                    ("G_repeat", "G_repeat_no_edges_cf"),
                    (
                        "O_label_sparse",
                        "O_label_sparse_no_edges_cf",
                    ),
                ):
                    probabilities[counterfactual_name] = (
                        predict_relational_probabilities(
                            results[full_model],
                            inputs.features,
                            inputs.local_probabilities,
                            empty_edge_indices(),
                            list(RELATION_NAMES),
                            device=device,
                        )
                    )
                _record_evaluation(
                    metric_records,
                    prediction_store,
                    test_truth,
                    probabilities,
                    labels,
                    evaluation_mask,
                    seed=seed,
                    training_scenario=scenario_name,
                    condition=condition,
                    realization=realization,
                    train_fraction=fraction,
                    checkpoint_condition=checkpoint_condition,
                    primary_metric=(
                        "accuracy"
                        if checkpoint_condition != "clean_global"
                        else "macro_f1"
                    ),
                    document_ids=test_document_ids,
                )

                if condition in {
                    "supervision_1.00",
                    "target_surface",
                    "component_surface",
                    "target_prior",
                    "component_prior",
                }:
                    for full_model in ("G_repeat", "O_label_sparse"):
                        full_metrics = _evaluate_probabilities(
                            test_truth,
                            probabilities[full_model],
                            labels,
                            evaluation_mask,
                        )
                        no_edge_name = (
                            "G_repeat_no_edges_cf"
                            if full_model == "G_repeat"
                            else "O_label_sparse_no_edges_cf"
                        )
                        no_edge_metrics = _evaluate_probabilities(
                            test_truth,
                            probabilities[no_edge_name],
                            labels,
                            evaluation_mask,
                        )
                        counterfactual_records.append(
                            {
                                "seed": seed,
                                "training_scenario": scenario_name,
                                "condition": condition,
                                "realization": realization,
                                "model": full_model,
                                "intervention": "no_edges",
                                "topology_seed": None,
                                "support": int(evaluation_mask.sum()),
                                "full_accuracy": full_metrics["accuracy"],
                                "intervention_accuracy": no_edge_metrics[
                                    "accuracy"
                                ],
                                "full_minus_intervention_accuracy": (
                                    float(full_metrics["accuracy"])
                                    - float(no_edge_metrics["accuracy"])
                                ),
                                "full_macro_f1": full_metrics["macro_f1"],
                                "intervention_macro_f1": no_edge_metrics[
                                    "macro_f1"
                                ],
                                "full_minus_intervention_macro_f1": (
                                    float(full_metrics["macro_f1"])
                                    - float(no_edge_metrics["macro_f1"])
                                ),
                            }
                        )
                        for topology_seed in topology_seeds:
                            randomized_edges = randomized_test_edges[
                                full_model
                            ][topology_seed]
                            randomized = predict_relational_probabilities(
                                results[full_model],
                                inputs.features,
                                inputs.local_probabilities,
                                randomized_edges,
                                list(RELATION_NAMES),
                                device=device,
                            )
                            randomized_metrics = _evaluate_probabilities(
                                test_truth,
                                randomized,
                                labels,
                                evaluation_mask,
                            )
                            counterfactual_records.append(
                                {
                                    "seed": seed,
                                    "training_scenario": scenario_name,
                                    "condition": condition,
                                    "realization": realization,
                                    "model": full_model,
                                    "intervention": "degree_preserving_random",
                                    "topology_seed": topology_seed,
                                    "support": int(evaluation_mask.sum()),
                                    "full_accuracy": full_metrics["accuracy"],
                                    "intervention_accuracy": (
                                        randomized_metrics["accuracy"]
                                    ),
                                    "full_minus_intervention_accuracy": (
                                        float(full_metrics["accuracy"])
                                        - float(
                                            randomized_metrics["accuracy"]
                                        )
                                    ),
                                    "full_macro_f1": full_metrics["macro_f1"],
                                    "intervention_macro_f1": (
                                        randomized_metrics["macro_f1"]
                                    ),
                                    "full_minus_intervention_macro_f1": (
                                        float(full_metrics["macro_f1"])
                                        - float(
                                            randomized_metrics["macro_f1"]
                                        )
                                    ),
                                }
                            )
                return probabilities

            if mode == "identity":
                global_condition = f"supervision_{fraction:.2f}"
                global_probabilities = evaluate_condition(
                    condition=global_condition,
                    realization=0,
                    evidence_mode="identity",
                    evidence_mask=np.zeros(len(test_truth), dtype=bool),
                    evaluation_mask=np.ones(len(test_truth), dtype=bool),
                    checkpoint_condition="clean_global",
                )
                if fraction == 1.0:
                    clean_full_probabilities = global_probabilities
                    clean_full_validation = validation_probabilities
                    for corruption_seed, (
                        target_mask,
                        _,
                        _,
                    ) in test_masks.items():
                        evaluate_condition(
                            condition="clean_targets",
                            realization=corruption_seed,
                            evidence_mode="identity",
                            evidence_mask=np.zeros(
                                len(test_truth),
                                dtype=bool,
                            ),
                            evaluation_mask=target_mask,
                            checkpoint_condition="clean_targets",
                        )
            elif mode == "surface":
                for corruption_seed, (
                    target_mask,
                    component_mask,
                    _,
                ) in test_masks.items():
                    evaluate_condition(
                        condition="target_surface",
                        realization=corruption_seed,
                        evidence_mode="surface",
                        evidence_mask=target_mask,
                        evaluation_mask=target_mask,
                        checkpoint_condition="target_surface",
                    )
                    evaluate_condition(
                        condition="component_surface",
                        realization=corruption_seed,
                        evidence_mode="surface",
                        evidence_mask=component_mask,
                        evaluation_mask=target_mask,
                        checkpoint_condition="target_surface",
                    )
            elif mode == "prior":
                for corruption_seed, (
                    target_mask,
                    component_mask,
                    _,
                ) in test_masks.items():
                    evaluate_condition(
                        condition="target_prior",
                        realization=corruption_seed,
                        evidence_mode="prior",
                        evidence_mask=target_mask,
                        evaluation_mask=target_mask,
                        checkpoint_condition="target_prior",
                    )
                    evaluate_condition(
                        condition="component_prior",
                        realization=corruption_seed,
                        evidence_mode="prior",
                        evidence_mask=component_mask,
                        evaluation_mask=target_mask,
                        checkpoint_condition="target_prior",
                    )
            else:
                raise RuntimeError(f"unsupported scenario mode {mode!r}")

            del results
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        if clean_full_probabilities is None or clean_full_validation is None:
            raise RuntimeError("clean full-supervision probabilities are missing")
        clean_test_probabilities_by_seed[seed] = {
            model_name: probabilities.copy()
            for model_name, probabilities in clean_full_probabilities.items()
        }
        clean_validation_n0_by_seed[seed] = clean_full_validation[
            "N0_node_only"
        ].copy()

        prediction_payload: dict[str, object] = {
            "mention_ids": np.asarray(
                [mention.mention_id for mention in test_mentions]
            ),
            "document_ids": np.asarray(test_document_ids),
            "truth": test_truth,
            "config_checksum": np.asarray([checksum]),
            "source_checksum": np.asarray([code_checksum]),
        }
        for (
            prediction_seed,
            condition,
            realization,
            model,
        ), prediction in prediction_store.items():
            if prediction_seed != seed:
                continue
            key = (
                f"{condition}__r{realization}__"
                f"{model.lower()}_prediction"
            )
            prediction_payload[key] = prediction
        np.savez_compressed(
            seed_directory / "test_predictions.npz",
            **prediction_payload,
        )
        for local in local_by_fraction.values():
            del local.local_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    ensemble_validation_n0 = np.mean(
        np.stack(
            [
                clean_validation_n0_by_seed[seed]
                for seed in seeds
            ],
            axis=0,
        ),
        axis=0,
    )
    ensemble_test_n0 = np.mean(
        np.stack(
            [
                clean_test_probabilities_by_seed[seed]["N0_node_only"]
                for seed in seeds
            ],
            axis=0,
        ),
        axis=0,
    )
    cohorts, shared_cohort_metadata = _ambiguity_cohorts(
        flatten_mentions(train_documents),
        test_mentions,
        ensemble_validation_n0,
        ensemble_test_n0,
        minimum_surface_support=int(
            diagnostic.get("ambiguous_surface_minimum_support", 5)
        ),
        minimum_surface_entropy_bits=float(
            diagnostic.get(
                "ambiguous_surface_minimum_entropy_bits",
                0.5,
            )
        ),
        uncertainty_quantile=float(
            diagnostic.get("uncertainty_quantile", 0.75)
        ),
    )
    cohorts["confident"] = ~cohorts["uncertain"]
    cohorts["train_seen_unambiguous"] = (
        cohorts["seen_surface"] & ~cohorts["train_ambiguous"]
    )
    ambiguity_metadata.append(
        {
            "shared_across_optimization_seeds": True,
            "seeds_averaged_before_thresholding": seeds,
            **shared_cohort_metadata,
            "cohort_supports": {
                cohort: int(mask.sum())
                for cohort, mask in cohorts.items()
            },
        }
    )
    np.savez_compressed(
        output_directory / "cohort_masks.npz",
        mention_ids=np.asarray(
            [mention.mention_id for mention in test_mentions]
        ),
        **{
            cohort: mask.astype(np.uint8)
            for cohort, mask in cohorts.items()
        },
    )
    for cohort, mask in cohorts.items():
        if not mask.any():
            continue
        condition_name = f"cohort_{cohort}"
        evaluation_masks[(condition_name, 0)] = mask.copy()
        selected_document_count = len(
            {
                test_mentions[index].doc_id
                for index in np.flatnonzero(mask)
            }
        )
        for seed in seeds:
            clean_probabilities = clean_test_probabilities_by_seed[seed]
            for model_name, probabilities in clean_probabilities.items():
                prediction_store[
                    (seed, condition_name, 0, model_name)
                ] = probabilities.argmax(axis=1).astype(np.int16)
            for model_name in (
                "local",
                "N0_node_only",
                "G_repeat",
                "G_semantic_union",
                "O_label_sparse",
            ):
                probabilities = clean_probabilities[model_name]
                metrics = _evaluate_probabilities(
                    test_truth,
                    probabilities,
                    labels,
                    mask,
                )
                transition = _transition_summary(
                    test_truth,
                    clean_probabilities["N0_node_only"],
                    probabilities,
                    mask,
                )
                ambiguity_records.append(
                    {
                        "seed": seed,
                        "cohort": cohort,
                        "model": model_name,
                        "support": int(mask.sum()),
                        "documents": selected_document_count,
                        "powered": bool(
                            mask.sum() >= 200
                            and selected_document_count >= 10
                        ),
                        **MODEL_METADATA[model_name],
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        **transition,
                    }
                )

    save_json(output_directory / "target_selection.json", target_selection)

    summary_records = _summarize_records(
        metric_records,
        keys=(
            "training_scenario",
            "condition",
            "model",
        ),
    )
    write_csv(output_directory / "metrics_raw.csv", metric_records)
    write_csv(output_directory / "metrics_summary.csv", summary_records)
    write_csv(
        output_directory / "table_g1_oracle_clean.csv",
        [
            row
            for row in summary_records
            if row["condition"] == "supervision_1.00"
            and row["model"]
            in {
                "local",
                "N0_node_only",
                "G_repeat",
                "G_semantic_union",
                "O_pruned_union",
                "O_label_sparse",
                "AVG_repeat",
                "AVG_label_sparse",
                "GoldVote_repeat",
                "GoldVote_label_sparse",
                "G_repeat_no_edges_cf",
                "O_label_sparse_no_edges_cf",
            }
        ],
    )
    write_csv(
        output_directory / "table_g2_target_recovery.csv",
        [
            row
            for row in summary_records
            if row["condition"]
            in {
                "clean_targets",
                "target_surface",
                "component_surface",
                "target_prior",
                "component_prior",
            }
        ],
    )
    write_csv(
        output_directory / "table_g3_low_resource.csv",
        [
            row
            for row in summary_records
            if str(row["condition"]).startswith("supervision_")
            and row["model"]
            in {
                "local",
                "N0_node_only",
                "G_repeat",
                "O_label_sparse",
                "AVG_repeat",
                "AVG_label_sparse",
            }
        ],
    )
    ambiguity_summary = _summarize_records(
        ambiguity_records,
        keys=("cohort", "model"),
        metrics=(
            "accuracy",
            "macro_f1",
            "correction_rate",
            "harm_rate",
            "net_accuracy_change",
        ),
    )
    write_csv(output_directory / "ambiguity_raw.csv", ambiguity_records)
    write_csv(
        output_directory / "table_g4_ambiguity.csv",
        ambiguity_summary,
    )
    save_json(
        output_directory / "ambiguity_thresholds.json",
        ambiguity_metadata,
    )

    counterfactual_per_seed = _summarize_records(
        counterfactual_records,
        keys=("training_scenario", "condition", "model", "intervention"),
        metrics=(
            "full_accuracy",
            "intervention_accuracy",
            "full_minus_intervention_accuracy",
            "full_macro_f1",
            "intervention_macro_f1",
            "full_minus_intervention_macro_f1",
        ),
    )
    write_csv(
        output_directory / "counterfactual_raw.csv",
        counterfactual_records,
    )
    write_csv(
        output_directory / "table_g5_counterfactual.csv",
        counterfactual_per_seed,
    )
    write_csv(
        output_directory / "propagation_alpha_selection.csv",
        alpha_records,
    )

    contrasts = [
        {
            "family": "F1_oracle_headroom",
            "contrast": "O_label_sparse_minus_N0_clean",
            "metric": "macro_f1",
            "sesoi": 0.01,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "supervision_1.00",
                    "model": "O_label_sparse",
                },
                {
                    "coefficient": -1,
                    "condition": "supervision_1.00",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F1_oracle_headroom",
            "contrast": "AVG_label_sparse_minus_N0_clean",
            "metric": "macro_f1",
            "sesoi": 0.01,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "supervision_1.00",
                    "model": "AVG_label_sparse",
                },
                {
                    "coefficient": -1,
                    "condition": "supervision_1.00",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F1_oracle_headroom",
            "contrast": "GoldVote_label_sparse_minus_N0_clean",
            "metric": "macro_f1",
            "sesoi": 0.01,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "supervision_1.00",
                    "model": "GoldVote_label_sparse",
                },
                {
                    "coefficient": -1,
                    "condition": "supervision_1.00",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F1_oracle_headroom",
            "contrast": "O_label_sparse_minus_AVG_label_sparse_noninferiority",
            "metric": "macro_f1",
            "sesoi": 0.01,
            "noninferiority_margin": 0.01,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "supervision_1.00",
                    "model": "O_label_sparse",
                },
                {
                    "coefficient": -1,
                    "condition": "supervision_1.00",
                    "model": "AVG_label_sparse",
                },
            ],
        },
        {
            "family": "F2_natural_edges",
            "contrast": "G_repeat_minus_N0_clean",
            "metric": "macro_f1",
            "sesoi": 0.01,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "supervision_1.00",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "supervision_1.00",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F2_natural_edges",
            "contrast": "AVG_repeat_minus_G_repeat_architecture",
            "metric": "macro_f1",
            "sesoi": 0.01,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "supervision_1.00",
                    "model": "AVG_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "supervision_1.00",
                    "model": "G_repeat",
                },
            ],
        },
        {
            "family": "F2_natural_edges",
            "contrast": "G_semantic_union_minus_N0_clean",
            "metric": "macro_f1",
            "sesoi": 0.01,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "supervision_1.00",
                    "model": "G_semantic_union",
                },
                {
                    "coefficient": -1,
                    "condition": "supervision_1.00",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F2_natural_edges",
            "contrast": "O_pruned_union_minus_G_semantic_union",
            "metric": "macro_f1",
            "sesoi": 0.01,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "supervision_1.00",
                    "model": "O_pruned_union",
                },
                {
                    "coefficient": -1,
                    "condition": "supervision_1.00",
                    "model": "G_semantic_union",
                },
            ],
        },
        {
            "family": "F3_target_recovery",
            "contrast": "G_repeat_minus_N0_target_surface",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "target_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "target_surface",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F3_target_recovery",
            "contrast": "G_repeat_direct_messages_target_surface",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "target_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "target_surface",
                    "model": "G_repeat_no_edges_cf",
                },
            ],
        },
        {
            "family": "F3_target_recovery",
            "contrast": "O_label_sparse_minus_N0_target_surface",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "target_surface",
                    "model": "O_label_sparse",
                },
                {
                    "coefficient": -1,
                    "condition": "target_surface",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F3_target_recovery",
            "contrast": (
                "repeat_context_borrowing_target_minus_component_surface"
            ),
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "target_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "target_surface",
                    "model": "N0_node_only",
                },
                {
                    "coefficient": -1,
                    "condition": "component_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": 1,
                    "condition": "component_surface",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F3_target_recovery",
            "contrast": "repeat_gain_target_surface_minus_clean_targets",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "target_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "target_surface",
                    "model": "N0_node_only",
                },
                {
                    "coefficient": -1,
                    "condition": "clean_targets",
                    "model": "G_repeat",
                },
                {
                    "coefficient": 1,
                    "condition": "clean_targets",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F4_low_resource",
            "contrast": "G_repeat_minus_N0_supervision_010",
            "metric": "macro_f1",
            "sesoi": 0.01,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "supervision_0.10",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "supervision_0.10",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F4_low_resource",
            "contrast": (
                "repeat_gain_low_resource_mean_010_025_minus_full"
            ),
            "metric": "macro_f1",
            "sesoi": 0.01,
            "terms": [
                {
                    "coefficient": 0.5,
                    "condition": "supervision_0.10",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -0.5,
                    "condition": "supervision_0.10",
                    "model": "N0_node_only",
                },
                {
                    "coefficient": 0.5,
                    "condition": "supervision_0.25",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -0.5,
                    "condition": "supervision_0.25",
                    "model": "N0_node_only",
                },
                {
                    "coefficient": -1.0,
                    "condition": "supervision_1.00",
                    "model": "G_repeat",
                },
                {
                    "coefficient": 1.0,
                    "condition": "supervision_1.00",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F5_subgroup_interactions",
            "contrast": "repeat_gain_repeated_minus_singleton",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "cohort_repeated",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "cohort_repeated",
                    "model": "N0_node_only",
                },
                {
                    "coefficient": -1,
                    "condition": "cohort_singleton",
                    "model": "G_repeat",
                },
                {
                    "coefficient": 1,
                    "condition": "cohort_singleton",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "F5_subgroup_interactions",
            "contrast": "repeat_gain_uncertain_minus_confident",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "cohort_uncertain",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "cohort_uncertain",
                    "model": "N0_node_only",
                },
                {
                    "coefficient": -1,
                    "condition": "cohort_confident",
                    "model": "G_repeat",
                },
                {
                    "coefficient": 1,
                    "condition": "cohort_confident",
                    "model": "N0_node_only",
                },
            ],
        },
    ]
    bootstrap_rows = _crossed_bootstrap(
        test_truth,
        prediction_store,
        evaluation_masks,
        test_document_ids,
        document_sources,
        labels,
        contrasts,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    write_csv(
        output_directory / "table_g6_crossed_bootstrap.csv",
        bootstrap_rows,
    )
    save_json(
        output_directory / "crossed_bootstrap.json",
        bootstrap_rows,
    )
    write_csv(
        output_directory / "table_g7_graph_diagnostics.csv",
        graph_records,
    )
    save_json(
        output_directory / "graph_diagnostics.json",
        graph_records,
    )
    save_json(
        output_directory / "dataset_statistics.json",
        corpus_statistics(documents),
    )
    save_json(output_directory / "config.json", config)
    manifest = _manifest(
        config,
        device=device,
        checksum=checksum,
        code_checksum=code_checksum,
        embedding_cache=clean_cache_path,
    )
    manifest["clean_embedding_cache"] = str(clean_cache_path)
    manifest["surface_only_embedding_cache"] = str(surface_cache_path)
    manifest["diagnostic_protocol"] = str(
        project_root / "DIAGNOSTIC_PROTOCOL_UK.md"
    )
    save_json(output_directory / "diagnostic_manifest.json", manifest)
    summary = {
        "output_directory": str(output_directory),
        "device": device,
        "optimization_seeds": seeds,
        "corruption_seeds": corruption_seeds,
        "low_resource_fractions": fractions,
        "models_trained": len(graph_records),
        "metric_records": len(metric_records),
        "bootstrap_contrasts": len(bootstrap_rows),
        "config_checksum": checksum,
        "source_checksum": code_checksum,
        "clean_embedding_cache": str(clean_cache_path),
        "surface_only_embedding_cache": str(surface_cache_path),
    }
    save_json(output_directory / "summary.json", summary)
    return summary
