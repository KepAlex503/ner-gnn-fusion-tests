from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .metrics import classification_metrics
from .models import (
    GraphSAGEClassifier,
    LocalClassifier,
    RelationalGatedClassifier,
    parameter_count,
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def class_weights(labels: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    weights = np.zeros(class_count, dtype=np.float64)
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present])
    if present.any():
        weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def _as_float_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _as_long_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.long, device=device)


def fit_local_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    config: dict[str, object],
    seed: int,
    device: str,
) -> LocalClassifier:
    set_random_seed(seed)
    model = LocalClassifier(
        input_dimension=features.shape[1],
        class_count=class_count,
        hidden_dimension=int(config.get("hidden_dimension", 0)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.02)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    weights = _as_float_tensor(class_weights(labels, class_count), device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    x = _as_float_tensor(features, device)
    y = _as_long_tensor(labels, device)
    model.train()
    for _ in range(int(config.get("epochs", 80))):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(x), y)
        loss.backward()
        optimizer.step()
    return model


def predict_local_probabilities(
    model: LocalClassifier,
    features: np.ndarray,
    *,
    device: str,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for left in range(0, len(features), batch_size):
            x = _as_float_tensor(features[left : left + batch_size], device)
            rows.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def document_folds(
    document_ids: list[str],
    document_sources: dict[str, str],
    *,
    fold_count: int,
    seed: int,
) -> dict[str, int]:
    by_source: dict[str, list[str]] = {}
    for document_id in sorted(set(document_ids)):
        by_source.setdefault(document_sources[document_id], []).append(document_id)
    assignments: dict[str, int] = {}
    for source, source_ids in sorted(by_source.items()):
        random.Random(f"{seed}:{source}").shuffle(source_ids)
        for index, document_id in enumerate(source_ids):
            assignments[document_id] = index % fold_count
    return assignments


def out_of_fold_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    document_ids: list[str],
    document_sources: dict[str, str],
    *,
    class_count: int,
    fold_count: int,
    config: dict[str, object],
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, int]]:
    assignments = document_folds(
        document_ids,
        document_sources,
        fold_count=fold_count,
        seed=seed,
    )
    folds = np.asarray([assignments[document_id] for document_id in document_ids])
    probabilities = np.full(
        (len(features), class_count),
        np.nan,
        dtype=np.float32,
    )
    for fold in range(fold_count):
        train_mask = folds != fold
        held_out_mask = folds == fold
        if not held_out_mask.any() or not train_mask.any():
            raise ValueError(f"invalid out-of-fold split {fold}")
        model = fit_local_classifier(
            features[train_mask],
            labels[train_mask],
            class_count=class_count,
            config=config,
            seed=seed * 100 + fold,
            device=device,
        )
        probabilities[held_out_mask] = predict_local_probabilities(
            model,
            features[held_out_mask],
            device=device,
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("out-of-fold predictions are incomplete")
    return probabilities, assignments


@dataclass(slots=True)
class GraphTrainingResult:
    model: GraphSAGEClassifier
    best_epoch: int
    best_validation_macro_f1: float
    parameter_count: int


def _graph_probabilities(
    model: GraphSAGEClassifier,
    features: np.ndarray,
    edge_index: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits = model(
            _as_float_tensor(features, device),
            _as_long_tensor(edge_index, device),
        )
        return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)


def fit_graph_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_edge_index: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_edge_index: np.ndarray,
    *,
    labels: list[str],
    config: dict[str, object],
    seed: int,
    device: str,
) -> GraphTrainingResult:
    set_random_seed(seed)
    model = GraphSAGEClassifier(
        input_dimension=train_features.shape[1],
        hidden_dimension=int(config.get("hidden_dimension", 128)),
        class_count=len(labels),
        dropout=float(config.get("dropout", 0.25)),
        layer_normalization=bool(config.get("layer_normalization", True)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.003)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    loss_function = nn.CrossEntropyLoss(
        weight=_as_float_tensor(
            class_weights(train_labels, len(labels)),
            device,
        )
    )
    x_train = _as_float_tensor(train_features, device)
    y_train = _as_long_tensor(train_labels, device)
    train_edges = _as_long_tensor(train_edge_index, device)
    patience = int(config.get("patience", 25))
    maximum_epochs = int(config.get("epochs", 250))
    best_epoch = 0
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(x_train, train_edges), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(config.get("gradient_clip", 5.0)),
        )
        optimizer.step()
        validation_probabilities = _graph_probabilities(
            model,
            validation_features,
            validation_edge_index,
            device=device,
        )
        validation_prediction = validation_probabilities.argmax(axis=1)
        score = float(
            classification_metrics(
                validation_labels,
                validation_prediction,
                labels,
            )["macro_f1"]
        )
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("GraphSAGE training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return GraphTrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_macro_f1=best_score,
        parameter_count=parameter_count(model),
    )


def predict_graph_probabilities(
    result: GraphTrainingResult,
    features: np.ndarray,
    edge_index: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    return _graph_probabilities(
        result.model,
        features,
        edge_index,
        device=device,
    )


@dataclass(slots=True)
class RelationalTrainingResult:
    model: RelationalGatedClassifier
    best_epoch: int
    best_validation_macro_f1: float
    parameter_count: int
    best_validation_score: float
    validation_metric: str


def _edge_tensor_dictionary(
    edge_indices: dict[str, np.ndarray],
    relation_names: list[str],
    device: str,
) -> dict[str, torch.Tensor]:
    missing = set(relation_names) - set(edge_indices)
    if missing:
        raise ValueError(f"missing relational edge indices: {sorted(missing)}")
    return {
        relation: _as_long_tensor(edge_indices[relation], device)
        for relation in relation_names
    }


def _relational_probabilities(
    model: RelationalGatedClassifier,
    features: np.ndarray,
    local_probabilities: np.ndarray,
    edge_indices: dict[str, np.ndarray],
    relation_names: list[str],
    *,
    device: str,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits = model(
            _as_float_tensor(features, device),
            _edge_tensor_dictionary(edge_indices, relation_names, device),
            _as_float_tensor(local_probabilities, device),
        )
        return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)


def fit_relational_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_local_probabilities: np.ndarray,
    train_edge_indices: dict[str, np.ndarray],
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_local_probabilities: np.ndarray,
    validation_edge_indices: dict[str, np.ndarray],
    *,
    labels: list[str],
    relation_names: list[str],
    config: dict[str, object],
    seed: int,
    device: str,
    validation_mask: np.ndarray | None = None,
    validation_metric: str = "macro_f1",
) -> RelationalTrainingResult:
    if validation_metric not in {"macro_f1", "accuracy"}:
        raise ValueError(
            "validation_metric must be either 'macro_f1' or 'accuracy'"
        )
    if validation_mask is None:
        validation_mask_array = np.ones(len(validation_labels), dtype=bool)
    else:
        validation_mask_array = np.asarray(validation_mask, dtype=bool)
        if validation_mask_array.shape != validation_labels.shape:
            raise ValueError("validation_mask and validation_labels differ")
        if not validation_mask_array.any():
            raise ValueError("validation_mask selects no examples")
    set_random_seed(seed)
    model = RelationalGatedClassifier(
        input_dimension=train_features.shape[1],
        hidden_dimension=int(config.get("hidden_dimension", 128)),
        class_count=len(labels),
        relation_names=relation_names,
        layers=int(config.get("layers", 1)),
        dropout=float(config.get("dropout", 0.25)),
        layer_normalization=bool(config.get("layer_normalization", True)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.003)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    loss_function = nn.CrossEntropyLoss(
        weight=_as_float_tensor(
            class_weights(train_labels, len(labels)),
            device,
        )
    )
    train_x = _as_float_tensor(train_features, device)
    train_y = _as_long_tensor(train_labels, device)
    train_local = _as_float_tensor(train_local_probabilities, device)
    train_edges = _edge_tensor_dictionary(
        train_edge_indices,
        relation_names,
        device,
    )
    patience = int(config.get("patience", 25))
    maximum_epochs = int(config.get("epochs", 250))
    best_epoch = 0
    best_score = -1.0
    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(
            model(train_x, train_edges, train_local),
            train_y,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(config.get("gradient_clip", 5.0)),
        )
        optimizer.step()
        validation_probabilities = _relational_probabilities(
            model,
            validation_features,
            validation_local_probabilities,
            validation_edge_indices,
            relation_names,
            device=device,
        )
        validation_metrics = classification_metrics(
            validation_labels[validation_mask_array],
            validation_probabilities.argmax(axis=1)[validation_mask_array],
            labels,
            include_empty_classes=validation_metric == "macro_f1",
        )
        score = float(validation_metrics[validation_metric])
        if score > best_score + 1e-8:
            best_score = score
            best_macro_f1 = float(validation_metrics["macro_f1"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("relational training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return RelationalTrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_macro_f1=best_macro_f1,
        parameter_count=parameter_count(model),
        best_validation_score=best_score,
        validation_metric=validation_metric,
    )


def predict_relational_probabilities(
    result: RelationalTrainingResult,
    features: np.ndarray,
    local_probabilities: np.ndarray,
    edge_indices: dict[str, np.ndarray],
    relation_names: list[str],
    *,
    device: str,
) -> np.ndarray:
    return _relational_probabilities(
        result.model,
        features,
        local_probabilities,
        edge_indices,
        relation_names,
        device=device,
    )
