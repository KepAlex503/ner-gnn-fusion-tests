from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from .graph import normalize_surface
from .schema import Mention


def confusion_matrix(
    truth: np.ndarray,
    prediction: np.ndarray,
    class_count: int,
) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction shapes differ")
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    return matrix


def metrics_from_confusion(
    matrix: np.ndarray,
    labels: list[str],
    *,
    include_empty_classes: bool = True,
) -> dict[str, object]:
    matrix = np.asarray(matrix)
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
    active = np.ones(len(labels), dtype=bool)
    if not include_empty_classes:
        active = support > 0
    if not active.any():
        active = np.ones(len(labels), dtype=bool)
    total = float(matrix.sum())
    accuracy = float(true_positive.sum() / total) if total else 0.0
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision[active].mean()),
        "macro_recall": float(recall[active].mean()),
        "macro_f1": float(f1[active].mean()),
        # In single-label multiclass classification, micro-F1 equals accuracy.
        "micro_f1": accuracy,
        "support": int(total),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def classification_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    labels: list[str],
    *,
    include_empty_classes: bool = True,
) -> dict[str, object]:
    return metrics_from_confusion(
        confusion_matrix(truth, prediction, len(labels)),
        labels,
        include_empty_classes=include_empty_classes,
    )


def probability_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(truth):
        raise ValueError("probability matrix has an invalid shape")
    clipped = np.clip(probabilities, 1e-12, 1.0)
    negative_log_likelihood = -np.log(clipped[np.arange(len(truth)), truth]).mean()
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(truth)), truth] = 1.0
    brier = np.square(probabilities - one_hot).sum(axis=1).mean()
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == truth
    expected_calibration_error = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= boundaries[index]) & (
                confidence <= boundaries[index + 1]
            )
        else:
            mask = (confidence >= boundaries[index]) & (
                confidence < boundaries[index + 1]
            )
        if not mask.any():
            continue
        expected_calibration_error += (
            float(mask.mean())
            * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        )
    return {
        "negative_log_likelihood": float(negative_log_likelihood),
        "brier_score": float(brier),
        "ece_10_bins": float(expected_calibration_error),
    }


def repeated_form_predictions(
    mentions: list[Mention],
    local_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(mentions) != len(local_probabilities):
        raise ValueError("mentions and probability rows differ")
    grouped_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, mention in enumerate(mentions):
        grouped_indices[
            (mention.doc_id, normalize_surface(mention.text))
        ].append(index)
    probabilities = np.asarray(local_probabilities, dtype=np.float32).copy()
    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue
        mean_probability = probabilities[indices].mean(axis=0)
        probabilities[indices] = mean_probability
    return probabilities.argmax(axis=1), probabilities


def repetition_buckets(mentions: list[Mention]) -> np.ndarray:
    counts = Counter(
        (mention.doc_id, normalize_surface(mention.text)) for mention in mentions
    )
    buckets: list[str] = []
    for mention in mentions:
        count = counts[(mention.doc_id, normalize_surface(mention.text))]
        buckets.append("1" if count == 1 else "2" if count == 2 else "3+")
    return np.asarray(buckets)


def grouped_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: Iterable[str],
    labels: list[str],
) -> dict[str, dict[str, object]]:
    groups = np.asarray(list(groups))
    result: dict[str, dict[str, object]] = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        result[group] = classification_metrics(
            truth[mask],
            prediction[mask],
            labels,
            include_empty_classes=False,
        )
    return result


def _bootstrap_p_value(deltas: np.ndarray) -> float:
    non_positive = (np.count_nonzero(deltas <= 0.0) + 1) / (len(deltas) + 1)
    non_negative = (np.count_nonzero(deltas >= 0.0) + 1) / (len(deltas) + 1)
    return float(min(1.0, 2.0 * min(non_positive, non_negative)))


def paired_document_bootstrap(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    graph_prediction: np.ndarray,
    document_ids: list[str],
    labels: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    if not (
        len(truth)
        == len(baseline_prediction)
        == len(graph_prediction)
        == len(document_ids)
    ):
        raise ValueError("bootstrap arrays have different lengths")
    unique_documents = sorted(set(document_ids))
    document_confusions: list[tuple[np.ndarray, np.ndarray]] = []
    document_ids_array = np.asarray(document_ids)
    for document_id in unique_documents:
        mask = document_ids_array == document_id
        document_confusions.append(
            (
                confusion_matrix(
                    truth[mask],
                    baseline_prediction[mask],
                    len(labels),
                ),
                confusion_matrix(
                    truth[mask],
                    graph_prediction[mask],
                    len(labels),
                ),
            )
        )
    baseline_stack = np.stack([pair[0] for pair in document_confusions], axis=0)
    graph_stack = np.stack([pair[1] for pair in document_confusions], axis=0)
    generator = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        chosen = generator.integers(0, len(unique_documents), len(unique_documents))
        baseline = metrics_from_confusion(
            baseline_stack[chosen].sum(axis=0),
            labels,
        )["macro_f1"]
        graph = metrics_from_confusion(
            graph_stack[chosen].sum(axis=0),
            labels,
        )["macro_f1"]
        deltas[sample] = float(graph) - float(baseline)
    observed_baseline = classification_metrics(
        truth,
        baseline_prediction,
        labels,
    )["macro_f1"]
    observed_graph = classification_metrics(
        truth,
        graph_prediction,
        labels,
    )["macro_f1"]
    return {
        "unit": "document",
        "samples": samples,
        "seed": seed,
        "documents": len(unique_documents),
        "observed_delta_macro_f1": float(observed_graph)
        - float(observed_baseline),
        "bootstrap_mean_delta_macro_f1": float(deltas.mean()),
        "confidence_interval_95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "two_sided_bootstrap_p": _bootstrap_p_value(deltas),
    }
