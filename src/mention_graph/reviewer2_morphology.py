from __future__ import annotations

import importlib.metadata
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import corpus_statistics, documents_for_split
from .diagnostic_controls import as_single_relation_edge_indices
from .diagnostic_pipeline import (
    DIAGNOSTIC_SEEDS,
    PROPAGATION_ALPHA_GRID,
    SINGLE_RELATION,
    _crossed_bootstrap,
    _edge_regimes,
    _evidence_inputs,
    _prepare_local_views,
    _propagate_probabilities,
    _record_evaluation,
    _select_propagation_alpha,
    _summarize_records,
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
from .graph import canonical_pair, normalize_surface
from .relational import RELATION_NAMES, RelationalGraphSource, build_relational_source
from .reporting import save_json, write_csv
from .schema import Document, Mention
from .training import (
    fit_relational_classifier,
    predict_relational_probabilities,
)


WORD_OR_PUNCT_RE = re.compile(r"[^\W_]+|[^\w\s]", flags=re.UNICODE)


def create_ukrainian_analyzer() -> Any:
    try:
        import pymorphy3
    except ImportError as error:
        raise RuntimeError(
            "Reviewer 2 morphology control requires the 'morphology' extra"
        ) from error
    return pymorphy3.MorphAnalyzer(lang="uk")


def morphology_versions() -> dict[str, str]:
    return {
        "pymorphy3": importlib.metadata.version("pymorphy3"),
        "pymorphy3-dicts-uk": importlib.metadata.version(
            "pymorphy3-dicts-uk"
        ),
    }


def lemma_normalize_surface(text: str, analyzer: Any) -> str:
    normalized = normalize_surface(text)
    tokens = WORD_OR_PUNCT_RE.findall(normalized)
    result: list[str] = []
    for token in tokens:
        if any(character.isalpha() for character in token):
            parses = analyzer.parse(token)
            result.append(str(parses[0].normal_form) if parses else token)
        else:
            result.append(token)
    return " ".join(result)


def _directed_edge_index(edges: set[tuple[int, int]]) -> np.ndarray:
    directed = [
        pair
        for left, right in sorted(edges)
        for pair in ((left, right), (right, left))
    ]
    return (
        np.asarray(directed, dtype=np.int64).T
        if directed
        else np.empty((2, 0), dtype=np.int64)
    )


def _undirected_edges(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        canonical_pair(int(left), int(right))
        for left, right in np.asarray(edge_index, dtype=np.int64).T
        if int(left) != int(right)
    }


def build_lemma_edge_index(
    documents: list[Document],
    source: RelationalGraphSource,
    analyzer: Any,
) -> tuple[np.ndarray, dict[str, str]]:
    surface_to_lemma: dict[str, str] = {}
    edges: set[tuple[int, int]] = set()
    for document in sorted(documents, key=lambda item: item.doc_id):
        start, end = source.document_offsets[document.doc_id]
        mentions = document.sorted_mentions()
        if end - start != len(mentions):
            raise RuntimeError("lemma graph and relational source are misaligned")
        by_lemma: dict[str, list[int]] = defaultdict(list)
        for local_index, mention in enumerate(mentions):
            surface = normalize_surface(mention.text)
            lemma = surface_to_lemma.get(surface)
            if lemma is None:
                lemma = lemma_normalize_surface(mention.text, analyzer)
                surface_to_lemma[surface] = lemma
            by_lemma[lemma].append(start + local_index)
        for group in by_lemma.values():
            for left, right in combinations(group, 2):
                edges.add(canonical_pair(left, right))
    return _directed_edge_index(edges), surface_to_lemma


def _covered_nodes(edges: set[tuple[int, int]], count: int) -> np.ndarray:
    covered = np.zeros(count, dtype=bool)
    for left, right in edges:
        covered[left] = True
        covered[right] = True
    return covered


def _homogeneity(
    edges: set[tuple[int, int]],
    labels: np.ndarray,
) -> float:
    if not edges:
        return 0.0
    return float(
        np.mean([labels[left] == labels[right] for left, right in edges])
    )


def graph_statistics(
    *,
    split: str,
    mentions: list[Mention],
    labels: np.ndarray,
    label_names: list[str],
    exact_edge_index: np.ndarray,
    lemma_edge_index: np.ndarray,
    surface_to_lemma: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    exact = _undirected_edges(exact_edge_index)
    lemma = _undirected_edges(lemma_edge_index)
    exact_covered = _covered_nodes(exact, len(mentions))
    lemma_covered = _covered_nodes(lemma, len(mentions))
    newly_covered = lemma_covered & ~exact_covered
    added_edges = lemma - exact
    rows: list[dict[str, object]] = [
        {
            "split": split,
            "scope": "overall",
            "label": "ALL",
            "mentions": len(mentions),
            "exact_undirected_edges": len(exact),
            "lemma_undirected_edges": len(lemma),
            "added_lemma_edges": len(added_edges),
            "exact_covered_mentions": int(exact_covered.sum()),
            "exact_coverage": float(exact_covered.mean()),
            "lemma_covered_mentions": int(lemma_covered.sum()),
            "lemma_coverage": float(lemma_covered.mean()),
            "newly_covered_mentions": int(newly_covered.sum()),
            "exact_label_homogeneity": _homogeneity(exact, labels),
            "lemma_label_homogeneity": _homogeneity(lemma, labels),
            "added_edge_label_homogeneity": _homogeneity(
                added_edges,
                labels,
            ),
        }
    ]
    for label_id, label_name in enumerate(label_names):
        mask = labels == label_id
        support = int(mask.sum())
        rows.append(
            {
                "split": split,
                "scope": "class_coverage",
                "label": label_name,
                "mentions": support,
                "exact_undirected_edges": "",
                "lemma_undirected_edges": "",
                "added_lemma_edges": "",
                "exact_covered_mentions": int((exact_covered & mask).sum()),
                "exact_coverage": float(
                    (exact_covered & mask).sum() / max(1, support)
                ),
                "lemma_covered_mentions": int((lemma_covered & mask).sum()),
                "lemma_coverage": float(
                    (lemma_covered & mask).sum() / max(1, support)
                ),
                "newly_covered_mentions": int((newly_covered & mask).sum()),
                "exact_label_homogeneity": "",
                "lemma_label_homogeneity": "",
                "added_edge_label_homogeneity": "",
            }
        )

    examples: list[dict[str, object]] = []
    for left, right in sorted(added_edges):
        left_surface = normalize_surface(mentions[left].text)
        right_surface = normalize_surface(mentions[right].text)
        examples.append(
            {
                "split": split,
                "left_mention_id": mentions[left].mention_id,
                "right_mention_id": mentions[right].mention_id,
                "document_id": mentions[left].doc_id,
                "left_surface": mentions[left].text,
                "right_surface": mentions[right].text,
                "left_normalized_surface": left_surface,
                "right_normalized_surface": right_surface,
                "lemma_sequence": surface_to_lemma[left_surface],
                "left_label": label_names[int(labels[left])],
                "right_label": label_names[int(labels[right])],
                "same_label": bool(labels[left] == labels[right]),
            }
        )
    return rows, examples


def _model_edges(
    source: RelationalGraphSource,
    lemma_edge_index: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    exact = _edge_regimes(source, seed=0)
    return {
        "N0_node_only": exact["N0_node_only"],
        "G_repeat": exact["G_repeat"],
        "G_lemma": as_single_relation_edge_indices(
            lemma_edge_index,
            relation=SINGLE_RELATION,
        ),
    }


def _contrasts(sesoi: float) -> list[dict[str, object]]:
    pairs = [
        (
            "R2_morphology_primary",
            "G_lemma_minus_G_repeat",
            "G_lemma",
            "G_repeat",
        ),
        (
            "R2_morphology_primary",
            "G_lemma_minus_N0",
            "G_lemma",
            "N0_node_only",
        ),
        (
            "R2_morphology_secondary_macro_f1",
            "AVG_lemma_minus_N0",
            "AVG_lemma",
            "N0_node_only",
        ),
        (
            "R2_morphology_secondary_macro_f1",
            "G_lemma_minus_AVG_lemma",
            "G_lemma",
            "AVG_lemma",
        ),
    ]
    result: list[dict[str, object]] = []
    for family, name, candidate, reference in pairs:
        result.append(
            {
                "family": family,
                "contrast": name,
                "metric": "macro_f1",
                "sesoi": sesoi,
                "terms": [
                    {
                        "coefficient": 1,
                        "condition": "clean_all",
                        "model": candidate,
                    },
                    {
                        "coefficient": -1,
                        "condition": "clean_all",
                        "model": reference,
                    },
                ],
            }
        )
    for name, candidate, reference in (
        ("G_lemma_minus_G_repeat_accuracy", "G_lemma", "G_repeat"),
        ("G_lemma_minus_N0_accuracy", "G_lemma", "N0_node_only"),
        ("AVG_lemma_minus_N0_accuracy", "AVG_lemma", "N0_node_only"),
        ("G_lemma_minus_AVG_lemma_accuracy", "G_lemma", "AVG_lemma"),
    ):
        result.append(
            {
                "family": "R2_morphology_secondary_accuracy",
                "contrast": name,
                "metric": "accuracy",
                "sesoi": 0.005,
                "terms": [
                    {
                        "coefficient": 1,
                        "condition": "clean_all",
                        "model": candidate,
                    },
                    {
                        "coefficient": -1,
                        "condition": "clean_all",
                        "model": reference,
                    },
                ],
            }
        )
    return result


def run_reviewer2_morphology(
    config: dict[str, Any],
    *,
    project_root: Path,
    output_directory: Path,
) -> dict[str, object]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory {output_directory}"
        )
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

    settings = config["reviewer2_morphology"]
    versions = morphology_versions()
    expected_versions = {
        "pymorphy3": str(settings["pymorphy3_version"]),
        "pymorphy3-dicts-uk": str(
            settings["pymorphy3_dicts_uk_version"]
        ),
    }
    if versions != expected_versions:
        raise RuntimeError(
            f"morphology versions differ: {versions} != {expected_versions}"
        )
    analyzer = create_ukrainian_analyzer()
    seeds = [
        int(value)
        for value in settings.get("optimization_seeds", DIAGNOSTIC_SEEDS)
    ]
    alpha_grid = [
        float(value)
        for value in settings.get(
            "propagation_alpha_grid",
            PROPAGATION_ALPHA_GRID,
        )
    ]
    bootstrap_samples = int(settings.get("bootstrap_samples", 20000))
    bootstrap_seed = int(settings.get("bootstrap_seed", 27182818))
    sesoi = float(settings.get("practical_threshold_macro_f1", 0.01))
    near_threshold = int(config["graph"].get("near_threshold_tokens", 50))
    alias_threshold = float(config["diagnostics"].get("alias_threshold", 0.9))
    fold_count = int(config["training"].get("oof_folds", 5))
    local_config = config["training"]["local"]
    model_config = config["diagnostics"]["model"]

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
    clean_lookup = _embedding_lookup(clean_ids, clean_embeddings)

    validation_mentions, validation_embeddings = _split_embeddings(
        validation_documents,
        clean_lookup,
    )
    test_mentions, test_embeddings = _split_embeddings(
        test_documents,
        clean_lookup,
    )
    validation_truth = _labels_array(validation_mentions, label_to_id)
    test_truth = _labels_array(test_mentions, label_to_id)
    document_sources = {
        document.doc_id: document.source for document in documents
    }
    test_document_ids = [mention.doc_id for mention in test_mentions]

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
    validation_lemma_index, validation_lemmas = build_lemma_edge_index(
        validation_documents,
        validation_source,
        analyzer,
    )
    test_lemma_index, test_lemmas = build_lemma_edge_index(
        test_documents,
        test_source,
        analyzer,
    )
    validation_edges = _model_edges(validation_source, validation_lemma_index)
    test_edges = _model_edges(test_source, test_lemma_index)

    graph_rows: list[dict[str, object]] = []
    morphology_examples: list[dict[str, object]] = []
    for split, mentions, truth, edges, lemma_index, lemmas in (
        (
            "validation",
            validation_mentions,
            validation_truth,
            validation_edges,
            validation_lemma_index,
            validation_lemmas,
        ),
        (
            "test",
            test_mentions,
            test_truth,
            test_edges,
            test_lemma_index,
            test_lemmas,
        ),
    ):
        rows, examples = graph_statistics(
            split=split,
            mentions=mentions,
            labels=truth,
            label_names=labels,
            exact_edge_index=edges["G_repeat"][SINGLE_RELATION],
            lemma_edge_index=lemma_index,
            surface_to_lemma=lemmas,
        )
        graph_rows.extend(rows)
        morphology_examples.extend(examples)

    metric_records: list[dict[str, object]] = []
    training_records: list[dict[str, object]] = []
    alpha_records: list[dict[str, object]] = []
    prediction_store: dict[tuple[int, str, int, str], np.ndarray] = {}
    all_test = np.ones(len(test_truth), dtype=bool)
    all_validation = np.ones(len(validation_truth), dtype=bool)
    no_validation_evidence = np.zeros(len(validation_truth), dtype=bool)
    no_test_evidence = np.zeros(len(test_truth), dtype=bool)
    evaluation_masks = {("clean_all", 0): all_test.copy()}
    trained_heads = 0

    for seed in seeds:
        seed_directory = output_directory / f"seed-{seed}"
        seed_directory.mkdir(parents=True, exist_ok=True)
        local = _prepare_local_views(
            train_documents,
            clean_lookup,
            clean_lookup,
            validation_embeddings,
            validation_embeddings,
            test_embeddings,
            test_embeddings,
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
        train_lemma_index, _ = build_lemma_edge_index(
            train_documents,
            local.train_source,
            analyzer,
        )
        train_edges = _model_edges(local.train_source, train_lemma_index)
        no_train_evidence = np.zeros(len(local.train_truth), dtype=bool)
        all_train = np.ones(len(local.train_truth), dtype=bool)
        train_inputs = _evidence_inputs(
            local.train_clean_embeddings,
            local.train_clean_embeddings,
            local.train_clean_probabilities,
            local.train_clean_probabilities,
            local.train_source,
            mode="identity",
            evidence_mask=no_train_evidence,
            evaluation_mask=all_train,
            train_prior=local.train_prior,
            degradation_indicator=False,
        )
        validation_inputs = _evidence_inputs(
            validation_embeddings,
            validation_embeddings,
            local.validation_clean_probabilities,
            local.validation_clean_probabilities,
            validation_source,
            mode="identity",
            evidence_mask=no_validation_evidence,
            evaluation_mask=all_validation,
            train_prior=local.train_prior,
            degradation_indicator=False,
        )
        test_inputs = _evidence_inputs(
            test_embeddings,
            test_embeddings,
            local.test_clean_probabilities,
            local.test_clean_probabilities,
            test_source,
            mode="identity",
            evidence_mask=no_test_evidence,
            evaluation_mask=all_test,
            train_prior=local.train_prior,
            degradation_indicator=False,
        )

        results: dict[str, object] = {}
        validation_probabilities: dict[str, np.ndarray] = {}
        for model_name in ("N0_node_only", "G_repeat", "G_lemma"):
            result = fit_relational_classifier(
                train_inputs.features,
                local.train_truth,
                train_inputs.local_probabilities,
                train_edges[model_name],
                validation_inputs.features,
                validation_truth,
                validation_inputs.local_probabilities,
                validation_edges[model_name],
                labels=labels,
                relation_names=list(RELATION_NAMES),
                config=model_config,
                seed=seed,
                device=device,
                validation_mask=all_validation,
                validation_metric="macro_f1",
            )
            results[model_name] = result
            trained_heads += 1
            validation_probabilities[model_name] = (
                predict_relational_probabilities(
                    result,
                    validation_inputs.features,
                    validation_inputs.local_probabilities,
                    validation_edges[model_name],
                    list(RELATION_NAMES),
                    device=device,
                )
            )
            training_records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "parameter_count": result.parameter_count,
                    "best_epoch": result.best_epoch,
                    "best_validation_macro_f1": result.best_validation_score,
                }
            )
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
                    "training_scenario": "intervention_free_morphology",
                    "seed": seed,
                    "config_checksum": checksum,
                    "source_checksum": code_checksum,
                    "best_epoch": result.best_epoch,
                    "validation_metric": result.validation_metric,
                    "best_validation_score": result.best_validation_score,
                },
                seed_directory / f"{model_name.lower()}_model.pt",
            )

        selected_alphas: dict[str, float] = {}
        for graph_name, model_name in (
            ("repeat", "G_repeat"),
            ("lemma", "G_lemma"),
        ):
            alpha, rows = _select_propagation_alpha(
                validation_truth,
                validation_probabilities["N0_node_only"],
                validation_edges[model_name][SINGLE_RELATION],
                labels,
                grid=alpha_grid,
                evaluation_mask=all_validation,
                update_mask=all_validation,
                allowed_neighbor_mask=all_validation,
                metric="macro_f1",
            )
            selected_alphas[graph_name] = alpha
            alpha_records.extend(
                {
                    "seed": seed,
                    "graph": graph_name,
                    "selected": abs(row["alpha"] - alpha) < 1e-12,
                    **row,
                }
                for row in rows
            )

        probabilities = {
            model_name: predict_relational_probabilities(
                result,
                test_inputs.features,
                test_inputs.local_probabilities,
                test_edges[model_name],
                list(RELATION_NAMES),
                device=device,
            )
            for model_name, result in results.items()
        }
        probabilities["AVG_repeat"] = _propagate_probabilities(
            probabilities["N0_node_only"],
            test_edges["G_repeat"][SINGLE_RELATION],
            alpha=selected_alphas["repeat"],
            update_mask=all_test,
            allowed_neighbor_mask=all_test,
        )
        probabilities["AVG_lemma"] = _propagate_probabilities(
            probabilities["N0_node_only"],
            test_edges["G_lemma"][SINGLE_RELATION],
            alpha=selected_alphas["lemma"],
            update_mask=all_test,
            allowed_neighbor_mask=all_test,
        )
        _record_evaluation(
            metric_records,
            prediction_store,
            test_truth,
            probabilities,
            labels,
            all_test,
            seed=seed,
            training_scenario="intervention_free_morphology",
            condition="clean_all",
            realization=0,
            train_fraction=1.0,
            checkpoint_condition="clean_all",
            primary_metric="macro_f1",
            document_ids=test_document_ids,
        )

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
            if prediction_seed == seed:
                prediction_payload[
                    f"{condition}__r{realization}__{model.lower()}_prediction"
                ] = prediction
        np.savez_compressed(
            seed_directory / "test_predictions.npz",
            **prediction_payload,
        )
        del local.local_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    bootstrap_rows = _crossed_bootstrap(
        test_truth,
        prediction_store,
        evaluation_masks,
        test_document_ids,
        document_sources,
        labels,
        _contrasts(sesoi),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    summary_records = _summarize_records(
        metric_records,
        keys=("training_scenario", "condition", "model"),
    )
    write_csv(output_directory / "metrics_raw.csv", metric_records)
    write_csv(output_directory / "metrics_summary.csv", summary_records)
    write_csv(output_directory / "graph_statistics.csv", graph_rows)
    write_csv(
        output_directory / "morphology_added_edge_examples.csv",
        morphology_examples,
    )
    write_csv(
        output_directory / "graph_training_diagnostics.csv",
        training_records,
    )
    write_csv(
        output_directory / "propagation_alpha_selection.csv",
        alpha_records,
    )
    write_csv(
        output_directory / "table_r2_morphology_performance.csv",
        summary_records,
    )
    write_csv(
        output_directory / "table_r2_morphology_contrasts.csv",
        bootstrap_rows,
    )
    save_json(output_directory / "crossed_bootstrap.json", bootstrap_rows)
    save_json(output_directory / "graph_statistics.json", graph_rows)
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
    manifest.update(
        {
            "control": "reviewer2_morphological_edge_rule",
            "morphology_versions": versions,
            "normalization": (
                "existing surface normalization followed by token-level "
                "highest-ranked Ukrainian normal_form"
            ),
            "protocol": str(
                project_root.parent.parent
                / "revision"
                / "REVIEWER2_MORPHOLOGICAL_EDGE_PROTOCOL.md"
            ),
            "trained_relational_heads": trained_heads,
        }
    )
    save_json(output_directory / "run_manifest.json", manifest)

    summary = {
        "output_directory": str(output_directory),
        "config_checksum": checksum,
        "source_checksum": code_checksum,
        "device": device,
        "optimization_seeds": seeds,
        "trained_relational_heads": trained_heads,
        "bootstrap_contrasts": len(bootstrap_rows),
        "embedding_cache": str(clean_cache_path),
        "morphology_versions": versions,
    }
    save_json(output_directory / "summary.json", summary)
    return summary
