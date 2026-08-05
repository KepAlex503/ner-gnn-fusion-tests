from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import (
    corpus_statistics,
    documents_for_split,
    duplicate_diagnostics,
    flatten_mentions,
    load_neruk,
    validate_document_splits,
)
from .encoding import encode_with_cache
from .graph import GraphBatch, build_edge_sets, build_graph_batch, normalize_surface
from .metrics import (
    classification_metrics,
    grouped_metrics,
    paired_document_bootstrap,
    probability_metrics,
    repeated_form_predictions,
    repetition_buckets,
)
from .models import parameter_count
from .reporting import (
    save_json,
    write_dataset_table,
    write_main_tables,
    write_repetition_table,
    write_robustness_table,
)
from .schema import Document, Mention, NER_UK_LABELS
from .synthetic import SYNTHETIC_LABELS, generate_synthetic_corpus
from .training import (
    fit_graph_classifier,
    fit_local_classifier,
    out_of_fold_probabilities,
    predict_graph_probabilities,
    predict_local_probabilities,
)


EDGE_CONFIGURATIONS = ("none", "sent", "repeat", "near", "all", "random")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "dataset", "encoder", "training", "graph"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"{path}: missing configuration keys {sorted(missing)}")
    return config


def config_checksum(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def source_checksum(project_root: Path) -> str:
    digest = hashlib.sha256()
    candidates = [
        project_root / "pyproject.toml",
        *sorted((project_root / "src").rglob("*.py")),
        *sorted((project_root / "configs").glob("*.json")),
    ]
    for path in candidates:
        if not path.exists():
            continue
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def load_experiment_documents(
    config: dict[str, Any],
    project_root: Path,
) -> list[Document]:
    dataset = config["dataset"]
    kind = dataset["kind"]
    if kind == "synthetic":
        documents = generate_synthetic_corpus(
            train_documents=int(dataset.get("train_documents", 80)),
            validation_documents=int(dataset.get("validation_documents", 20)),
            test_documents=int(dataset.get("test_documents", 30)),
            seed=int(dataset.get("seed", 2026)),
        )
    elif kind == "neruk":
        raw_root = Path(str(dataset.get("root", "data/external/ner-uk")))
        root = raw_root if raw_root.is_absolute() else project_root / raw_root
        documents = load_neruk(
            root,
            validation_fraction=float(dataset.get("validation_fraction", 0.15)),
            split_seed=int(dataset.get("split_seed", 2026)),
        )
    else:
        raise ValueError(f"unknown dataset kind {kind!r}")
    validate_document_splits(documents)
    return documents


def _labels_for_experiment(
    documents: list[Document],
    config: dict[str, Any],
) -> list[str]:
    if "labels" in config:
        labels = [str(label) for label in config["labels"]]
    elif config["dataset"]["kind"] == "neruk":
        labels = list(NER_UK_LABELS)
    elif config["dataset"]["kind"] == "synthetic":
        labels = list(SYNTHETIC_LABELS)
    else:
        labels = sorted(
            {mention.label for document in documents for mention in document.mentions}
        )
    observed = {mention.label for document in documents for mention in document.mentions}
    missing = observed - set(labels)
    if missing:
        raise ValueError(f"labels omitted from configuration: {sorted(missing)}")
    return labels


def _embedding_lookup(
    mention_ids: list[str],
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(mention_ids) != len(embeddings):
        raise ValueError("embedding identifiers and rows differ")
    return {
        mention_id: embeddings[index]
        for index, mention_id in enumerate(mention_ids)
    }


def _split_embeddings(
    documents: list[Document],
    lookup: dict[str, np.ndarray],
) -> tuple[list[Mention], np.ndarray]:
    mentions = flatten_mentions(documents)
    return mentions, np.stack(
        [lookup[mention.mention_id] for mention in mentions],
        axis=0,
    ).astype(np.float32)


def _labels_array(
    mentions: list[Mention],
    label_to_id: dict[str, int],
) -> np.ndarray:
    return np.asarray(
        [label_to_id[mention.label] for mention in mentions],
        dtype=np.int64,
    )


def _assert_graph_alignment(
    batch: GraphBatch,
    mentions: list[Mention],
) -> None:
    expected = [mention.mention_id for mention in mentions]
    if batch.mention_ids != expected:
        raise RuntimeError("graph node order differs from embedding/probability order")


def _graph_features(
    embeddings: np.ndarray,
    local_probabilities: np.ndarray,
    batch: GraphBatch,
) -> np.ndarray:
    if not (
        len(embeddings)
        == len(local_probabilities)
        == len(batch.structural_features)
    ):
        raise ValueError("graph feature components have different row counts")
    return np.concatenate(
        [embeddings, local_probabilities, batch.structural_features],
        axis=1,
    ).astype(np.float32)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _save_seed_predictions(
    path: Path,
    *,
    mention_ids: list[str],
    document_ids: list[str],
    truth: np.ndarray,
    local_probabilities: np.ndarray,
    repeated_probabilities: np.ndarray,
    graph_probabilities: dict[str, np.ndarray],
    checksum: str,
    code_checksum: str,
) -> None:
    payload: dict[str, object] = {
        "mention_ids": np.asarray(mention_ids),
        "document_ids": np.asarray(document_ids),
        "truth": truth,
        "local_probabilities": local_probabilities,
        "repeat_consistency_probabilities": repeated_probabilities,
        "config_checksum": np.asarray([checksum]),
        "source_checksum": np.asarray([code_checksum]),
    }
    for configuration, probabilities in graph_probabilities.items():
        payload[f"graph_{configuration}_probabilities"] = probabilities
    np.savez_compressed(path, **payload)


def _top_probabilities(
    probabilities: np.ndarray,
    labels: list[str],
    limit: int = 4,
) -> list[dict[str, object]]:
    ordered = np.argsort(probabilities)[::-1][:limit]
    return [
        {"label": labels[index], "probability": float(probabilities[index])}
        for index in ordered
    ]


def _qualitative_examples(
    test_documents: list[Document],
    test_mentions: list[Mention],
    graph_batch: GraphBatch,
    truth: np.ndarray,
    local_probabilities: np.ndarray,
    graph_probabilities: np.ndarray,
    labels: list[str],
    *,
    near_threshold_tokens: int,
) -> list[dict[str, object]]:
    local_prediction = local_probabilities.argmax(axis=1)
    graph_prediction = graph_probabilities.argmax(axis=1)
    categories: dict[str, list[int]] = {
        "local_error_corrected_by_graph": np.flatnonzero(
            (local_prediction != truth) & (graph_prediction == truth)
        ).tolist(),
        "correct_local_prediction_harmed_by_graph": np.flatnonzero(
            (local_prediction == truth) & (graph_prediction != truth)
        ).tolist(),
        "unchanged_error": np.flatnonzero(
            (local_prediction != truth) & (graph_prediction != truth)
        ).tolist(),
    }

    repeated_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, mention in enumerate(test_mentions):
        repeated_groups[
            (mention.doc_id, normalize_surface(mention.text))
        ].append(index)
    conflicting: list[int] = []
    for indices in repeated_groups.values():
        if len(indices) > 1 and len(set(local_prediction[indices].tolist())) > 1:
            conflicting.extend(indices)
    categories["conflicting_repeat_predictions"] = conflicting

    neighbor_map: dict[int, set[int]] = defaultdict(set)
    for source, target in graph_batch.edge_index.T.tolist():
        neighbor_map[int(source)].add(int(target))

    pair_types: dict[tuple[int, int], list[str]] = defaultdict(list)
    for document in test_documents:
        offset, _ = graph_batch.document_offsets[document.doc_id]
        typed = build_edge_sets(
            document,
            near_threshold_tokens=near_threshold_tokens,
        )
        for edge_type, edges in typed.items():
            for left, right in edges:
                pair_types[(offset + left, offset + right)].append(edge_type)

    document_by_id = {document.doc_id: document for document in test_documents}
    examples: list[dict[str, object]] = []
    used: set[int] = set()
    for category, candidates in categories.items():
        ranked = sorted(
            candidates,
            key=lambda index: abs(
                float(graph_probabilities[index, truth[index]])
                - float(local_probabilities[index, truth[index]])
            ),
            reverse=True,
        )
        chosen = next((index for index in ranked if index not in used), None)
        if chosen is None:
            continue
        used.add(chosen)
        mention = test_mentions[chosen]
        document = document_by_id[mention.doc_id]
        neighbors: list[dict[str, object]] = []
        for neighbor_index in sorted(neighbor_map.get(chosen, set())):
            neighbor = test_mentions[neighbor_index]
            pair = (
                min(chosen, neighbor_index),
                max(chosen, neighbor_index),
            )
            neighbors.append(
                {
                    "mention_id": neighbor.mention_id,
                    "text": neighbor.text,
                    "gold": labels[truth[neighbor_index]],
                    "local": labels[local_prediction[neighbor_index]],
                    "edge_types": sorted(pair_types.get(pair, [])),
                }
            )
        examples.append(
            {
                "category": category,
                "mention_id": mention.mention_id,
                "document_id": mention.doc_id,
                "mention": mention.text,
                "context": document.text[
                    mention.sentence_start : mention.sentence_end
                ],
                "gold": labels[truth[chosen]],
                "local_prediction": labels[local_prediction[chosen]],
                "graph_prediction": labels[graph_prediction[chosen]],
                "local_probabilities": _top_probabilities(
                    local_probabilities[chosen],
                    labels,
                ),
                "graph_probabilities": _top_probabilities(
                    graph_probabilities[chosen],
                    labels,
                ),
                "neighbors": neighbors,
            }
        )
    return examples


def _manifest(
    config: dict[str, Any],
    *,
    device: str,
    checksum: str,
    code_checksum: str,
    embedding_cache: Path,
) -> dict[str, object]:
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unavailable"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_checksum_sha256": checksum,
        "source_checksum_sha256": code_checksum,
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers_version,
        "device": device,
        "gpu": gpu,
        "embedding_cache": str(embedding_cache),
    }


def run_experiment(
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
    if not train_documents or not validation_documents or not test_documents:
        raise ValueError("train, validation, and test must all contain documents")

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
    embedding_lookup = _embedding_lookup(all_ids, all_embeddings)
    train_mentions, train_embeddings = _split_embeddings(
        train_documents,
        embedding_lookup,
    )
    validation_mentions, validation_embeddings = _split_embeddings(
        validation_documents,
        embedding_lookup,
    )
    test_mentions, test_embeddings = _split_embeddings(
        test_documents,
        embedding_lookup,
    )
    train_truth = _labels_array(train_mentions, label_to_id)
    validation_truth = _labels_array(validation_mentions, label_to_id)
    test_truth = _labels_array(test_mentions, label_to_id)
    document_sources = {document.doc_id: document.source for document in documents}
    seeds = [int(seed) for seed in config["training"].get("seeds", [17, 29, 43])]
    local_config = config["training"]["local"]
    graph_config = config["training"]["graph"]
    oof_folds = int(config["training"].get("oof_folds", 5))
    near_threshold = int(config["graph"].get("near_threshold_tokens", 50))
    configurations = [
        str(item)
        for item in config["graph"].get(
            "configurations",
            list(EDGE_CONFIGURATIONS),
        )
    ]
    unknown_configurations = set(configurations) - set(EDGE_CONFIGURATIONS)
    if unknown_configurations:
        raise ValueError(
            f"unknown graph configurations: {sorted(unknown_configurations)}"
        )
    if "all" not in configurations:
        raise ValueError("the main GraphSAGE configuration 'all' is required")

    save_json(output_directory / "config.json", config)
    dataset_statistics = corpus_statistics(documents)
    save_json(
        output_directory / "dataset_statistics.json",
        dataset_statistics,
    )
    write_dataset_table(output_directory, dataset_statistics)
    duplicates = duplicate_diagnostics(
        documents,
        near_similarity_threshold=float(
            config["dataset"].get("near_duplicate_threshold", 0.9)
        ),
    )
    save_json(output_directory / "duplicate_diagnostics.json", duplicates)
    if duplicates["cross_split_exact_groups"]:
        raise ValueError("exact duplicate documents occur across data splits")

    run_records: list[dict[str, object]] = []
    repetition_records: list[dict[str, object]] = []
    robustness_records: list[dict[str, object]] = []
    bootstraps: list[dict[str, object]] = []
    graph_diagnostics: list[dict[str, object]] = []
    qualitative_examples: list[dict[str, object]] = []
    assignments_by_seed: dict[str, dict[str, int]] = {}

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
        assignments_by_seed[str(seed)] = fold_assignments
        local_model = fit_local_classifier(
            train_embeddings,
            train_truth,
            class_count=len(labels),
            config=local_config,
            seed=seed,
            device=device,
        )
        validation_local_probabilities = predict_local_probabilities(
            local_model,
            validation_embeddings,
            device=device,
        )
        test_local_probabilities = predict_local_probabilities(
            local_model,
            test_embeddings,
            device=device,
        )
        local_prediction = test_local_probabilities.argmax(axis=1)
        local_metrics = classification_metrics(
            test_truth,
            local_prediction,
            labels,
        )
        local_metrics.update(
            probability_metrics(test_truth, test_local_probabilities)
        )
        run_records.append(
            {"seed": seed, "model": "local", "metrics": local_metrics}
        )
        repeated_prediction, repeated_probabilities = repeated_form_predictions(
            test_mentions,
            test_local_probabilities,
        )
        repeated_metrics = classification_metrics(
            test_truth,
            repeated_prediction,
            labels,
        )
        repeated_metrics.update(
            probability_metrics(test_truth, repeated_probabilities)
        )
        run_records.append(
            {
                "seed": seed,
                "model": "repeat_consistency",
                "metrics": repeated_metrics,
            }
        )
        torch.save(
            {
                "state_dict": _cpu_state_dict(local_model),
                "input_dimension": train_embeddings.shape[1],
                "class_count": len(labels),
                "labels": labels,
                "config": local_config,
                "seed": seed,
                "config_checksum": checksum,
                "source_checksum": code_checksum,
                "parameter_count": parameter_count(local_model),
            },
            seed_directory / "local_model.pt",
        )

        graph_probabilities: dict[str, np.ndarray] = {}
        graph_results: dict[str, object] = {}
        test_batches: dict[str, GraphBatch] = {}
        for edge_configuration in configurations:
            train_batch = build_graph_batch(
                train_documents,
                label_to_id=label_to_id,
                configuration=edge_configuration,
                near_threshold_tokens=near_threshold,
                seed=seed,
            )
            validation_batch = build_graph_batch(
                validation_documents,
                label_to_id=label_to_id,
                configuration=edge_configuration,
                near_threshold_tokens=near_threshold,
                seed=seed,
            )
            test_batch = build_graph_batch(
                test_documents,
                label_to_id=label_to_id,
                configuration=edge_configuration,
                near_threshold_tokens=near_threshold,
                seed=seed,
            )
            _assert_graph_alignment(train_batch, train_mentions)
            _assert_graph_alignment(validation_batch, validation_mentions)
            _assert_graph_alignment(test_batch, test_mentions)
            train_graph_features = _graph_features(
                train_embeddings,
                oof_probabilities,
                train_batch,
            )
            validation_graph_features = _graph_features(
                validation_embeddings,
                validation_local_probabilities,
                validation_batch,
            )
            test_graph_features = _graph_features(
                test_embeddings,
                test_local_probabilities,
                test_batch,
            )
            training_result = fit_graph_classifier(
                train_graph_features,
                train_truth,
                train_batch.edge_index,
                validation_graph_features,
                validation_truth,
                validation_batch.edge_index,
                labels=labels,
                config=graph_config,
                seed=seed,
                device=device,
            )
            probabilities = predict_graph_probabilities(
                training_result,
                test_graph_features,
                test_batch.edge_index,
                device=device,
            )
            metrics = classification_metrics(
                test_truth,
                probabilities.argmax(axis=1),
                labels,
            )
            metrics.update(probability_metrics(test_truth, probabilities))
            model_name = f"graph_{edge_configuration}"
            run_records.append(
                {"seed": seed, "model": model_name, "metrics": metrics}
            )
            graph_probabilities[edge_configuration] = probabilities
            graph_results[edge_configuration] = training_result
            test_batches[edge_configuration] = test_batch
            graph_diagnostics.append(
                {
                    "seed": seed,
                    "configuration": edge_configuration,
                    "parameter_count": training_result.parameter_count,
                    "best_epoch": training_result.best_epoch,
                    "best_validation_macro_f1": (
                        training_result.best_validation_macro_f1
                    ),
                    "train_directed_edges": int(train_batch.edge_index.shape[1]),
                    "test_directed_edges": int(test_batch.edge_index.shape[1]),
                    "train_isolated_nodes": train_batch.isolated_nodes,
                    "test_isolated_nodes": test_batch.isolated_nodes,
                    "train_edge_counts": train_batch.edge_counts,
                    "test_edge_counts": test_batch.edge_counts,
                }
            )
            torch.save(
                {
                    "state_dict": _cpu_state_dict(training_result.model),
                    "input_dimension": train_graph_features.shape[1],
                    "labels": labels,
                    "config": graph_config,
                    "edge_configuration": edge_configuration,
                    "near_threshold_tokens": near_threshold,
                    "seed": seed,
                    "config_checksum": checksum,
                    "source_checksum": code_checksum,
                    "best_epoch": training_result.best_epoch,
                    "best_validation_macro_f1": (
                        training_result.best_validation_macro_f1
                    ),
                },
                seed_directory / f"graph_{edge_configuration}_model.pt",
            )

        graph_all_prediction = graph_probabilities["all"].argmax(axis=1)
        grouped_local = grouped_metrics(
            test_truth,
            local_prediction,
            repetition_buckets(test_mentions),
            labels,
        )
        grouped_graph = grouped_metrics(
            test_truth,
            graph_all_prediction,
            repetition_buckets(test_mentions),
            labels,
        )
        for bucket, metrics in grouped_local.items():
            repetition_records.append(
                {
                    "seed": seed,
                    "bucket": bucket,
                    "model": "local",
                    "metrics": metrics,
                }
            )
        for bucket, metrics in grouped_graph.items():
            repetition_records.append(
                {
                    "seed": seed,
                    "bucket": bucket,
                    "model": "graph_all",
                    "metrics": metrics,
                }
            )

        comparison_predictions: dict[str, np.ndarray] = {
            "local": local_prediction,
            "repeat_consistency": repeated_prediction,
            **{
                f"graph_{configuration}": probabilities.argmax(axis=1)
                for configuration, probabilities in graph_probabilities.items()
                if configuration != "all"
            },
        }
        bootstrap_samples = int(
            config.get("statistics", {}).get("bootstrap_samples", 2000)
        )
        bootstrap_seed = int(
            config.get("statistics", {}).get("bootstrap_seed", 8675309)
        )
        for comparison_index, (
            baseline_name,
            baseline_prediction,
        ) in enumerate(sorted(comparison_predictions.items())):
            bootstrap = paired_document_bootstrap(
                test_truth,
                baseline_prediction,
                graph_all_prediction,
                [mention.doc_id for mention in test_mentions],
                labels,
                samples=bootstrap_samples,
                seed=bootstrap_seed + seed * 100 + comparison_index,
            )
            bootstrap["model_a"] = baseline_name
            bootstrap["model_b"] = "graph_all"
            bootstrap["run_seed"] = seed
            bootstrap["multiplicity_note"] = (
                "Exploratory E2 p-values are uncorrected for multiple comparisons."
            )
            bootstraps.append(bootstrap)

        if not qualitative_examples:
            qualitative_examples = _qualitative_examples(
                test_documents,
                test_mentions,
                test_batches["all"],
                test_truth,
                test_local_probabilities,
                graph_probabilities["all"],
                labels,
                near_threshold_tokens=near_threshold,
            )

        robustness = config.get("robustness", {})
        if bool(robustness.get("enabled", False)):
            all_graph_result = graph_results["all"]
            all_test_batch = test_batches["all"]
            clean_graph_macro = float(
                classification_metrics(
                    test_truth,
                    graph_all_prediction,
                    labels,
                )["macro_f1"]
            )
            clean_local_macro = float(local_metrics["macro_f1"])
            for raw_condition in robustness.get("conditions", []):
                condition = {
                    "kind": str(raw_condition["kind"]),
                    "level": float(raw_condition["level"]),
                    "seed": int(raw_condition.get("seed", 2026)),
                }
                noisy_ids, noisy_embeddings, _ = encode_with_cache(
                    test_documents,
                    encoder=config["encoder"],
                    cache_root=cache_root,
                    model_cache=model_cache,
                    device=device,
                    noise=condition,
                )
                expected_ids = [mention.mention_id for mention in test_mentions]
                if noisy_ids != expected_ids:
                    raise RuntimeError("noisy embedding order differs from test mentions")
                noisy_local_probabilities = predict_local_probabilities(
                    local_model,
                    noisy_embeddings,
                    device=device,
                )
                noisy_local_prediction = noisy_local_probabilities.argmax(axis=1)
                noisy_graph_features = _graph_features(
                    noisy_embeddings,
                    noisy_local_probabilities,
                    all_test_batch,
                )
                noisy_graph_probabilities = predict_graph_probabilities(
                    all_graph_result,
                    noisy_graph_features,
                    all_test_batch.edge_index,
                    device=device,
                )
                for model_name, prediction, clean_macro in (
                    (
                        "local",
                        noisy_local_prediction,
                        clean_local_macro,
                    ),
                    (
                        "graph_all",
                        noisy_graph_probabilities.argmax(axis=1),
                        clean_graph_macro,
                    ),
                ):
                    robustness_records.append(
                        {
                            "seed": seed,
                            "noise_kind": condition["kind"],
                            "noise_level": condition["level"],
                            "model": model_name,
                            "clean_macro_f1": clean_macro,
                            "metrics": classification_metrics(
                                test_truth,
                                prediction,
                                labels,
                            ),
                        }
                    )

        _save_seed_predictions(
            seed_directory / "test_predictions.npz",
            mention_ids=[mention.mention_id for mention in test_mentions],
            document_ids=[mention.doc_id for mention in test_mentions],
            truth=test_truth,
            local_probabilities=test_local_probabilities,
            repeated_probabilities=repeated_probabilities,
            graph_probabilities=graph_probabilities,
            checksum=checksum,
            code_checksum=code_checksum,
        )
        del local_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_main_tables(output_directory, run_records, labels=labels)
    write_repetition_table(output_directory, repetition_records)
    write_robustness_table(output_directory, robustness_records)
    save_json(output_directory / "paired_bootstrap.json", bootstraps)
    save_json(output_directory / "metrics_detailed.json", run_records)
    save_json(
        output_directory / "graph_diagnostics.json",
        graph_diagnostics,
    )
    save_json(
        output_directory / "oof_document_folds.json",
        assignments_by_seed,
    )
    save_json(
        output_directory / "qualitative_examples.json",
        qualitative_examples,
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
        "dataset": dataset_statistics,
        "labels": labels,
        "seeds": seeds,
        "device": device,
        "config_checksum": checksum,
        "source_checksum": code_checksum,
        "runs": len(run_records),
    }
    save_json(output_directory / "summary.json", summary)
    return summary
