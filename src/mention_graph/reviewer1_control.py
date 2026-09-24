from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import corpus_statistics, documents_for_split
from .diagnostic_pipeline import (
    DEFAULT_CORRUPTION_SEEDS,
    DIAGNOSTIC_SEEDS,
    PROPAGATION_ALPHA_GRID,
    SINGLE_RELATION,
    _crossed_bootstrap,
    _edge_regimes,
    _evidence_inputs,
    _prepare_local_views,
    _propagate_probabilities,
    _record_evaluation,
    _repeat_masks,
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
from .relational import RELATION_NAMES, build_relational_source
from .reporting import save_json, write_csv
from .training import (
    fit_relational_classifier,
    predict_relational_probabilities,
)


INDICATOR_REGIMES = {
    "q_on": True,
    "q_off": False,
}


def _condition_name(indicator_regime: str, condition: str) -> str:
    return f"{indicator_regime}_{condition}"


def run_reviewer1_control(
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

    diagnostic = config["diagnostics"]
    control = config["reviewer1_control"]
    seeds = [
        int(value)
        for value in control.get("optimization_seeds", DIAGNOSTIC_SEEDS)
    ]
    corruption_seeds = [
        int(value)
        for value in control.get(
            "corruption_seeds",
            DEFAULT_CORRUPTION_SEEDS,
        )
    ]
    train_target_seed = int(control.get("train_target_seed", 314159))
    validation_target_seed = int(
        control.get("validation_target_seed", 161803)
    )
    alpha_grid = [
        float(value)
        for value in control.get(
            "propagation_alpha_grid",
            PROPAGATION_ALPHA_GRID,
        )
    ]
    bootstrap_samples = int(control.get("bootstrap_samples", 20000))
    bootstrap_seed = int(control.get("bootstrap_seed", 9071986))
    near_threshold = int(config["graph"].get("near_threshold_tokens", 50))
    alias_threshold = float(diagnostic.get("alias_threshold", 0.9))
    fold_count = int(config["training"].get("oof_folds", 5))
    local_config = config["training"]["local"]
    model_config = diagnostic["model"]

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
        noise={"kind": "context_dropout", "level": 1.0, "seed": 0},
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
    validation_edges = _edge_regimes(validation_source, seed=0)
    test_edges = _edge_regimes(test_source, seed=0)

    validation_target, _, validation_counts = _repeat_masks(
        validation_mentions,
        target_seed=validation_target_seed,
    )
    test_masks = {
        seed: _repeat_masks(test_mentions, target_seed=seed)
        for seed in corruption_seeds
    }
    target_selection = {
        "validation": {
            "seed": validation_target_seed,
            **validation_counts,
        },
        "test": {
            str(seed): {"seed": seed, **masks[2]}
            for seed, masks in test_masks.items()
        },
        "train": {},
    }

    metric_records: list[dict[str, object]] = []
    graph_records: list[dict[str, object]] = []
    alpha_records: list[dict[str, object]] = []
    prediction_store: dict[
        tuple[int, str, int, str],
        np.ndarray,
    ] = {}
    evaluation_masks: dict[tuple[str, int], np.ndarray] = {}
    for indicator_regime in INDICATOR_REGIMES:
        for corruption_seed, (target, _, _) in test_masks.items():
            for condition in (
                "clean_targets",
                "target_surface",
                "component_surface",
            ):
                evaluation_masks[
                    (
                        _condition_name(indicator_regime, condition),
                        corruption_seed,
                    )
                ] = target.copy()

    trained_heads = 0
    for seed in seeds:
        seed_directory = output_directory / f"seed-{seed}"
        seed_directory.mkdir(parents=True, exist_ok=True)
        local = _prepare_local_views(
            train_documents,
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
        train_target, _, train_counts = _repeat_masks(
            local.train_mentions,
            target_seed=train_target_seed,
        )
        target_selection["train"][str(seed)] = {
            "seed": train_target_seed,
            **train_counts,
        }
        train_edges = _edge_regimes(local.train_source, seed=seed)

        for indicator_regime, indicator_enabled in INDICATOR_REGIMES.items():
            train_inputs = _evidence_inputs(
                local.train_clean_embeddings,
                local.train_surface_embeddings,
                local.train_clean_probabilities,
                local.train_surface_probabilities,
                local.train_source,
                mode="surface",
                evidence_mask=train_target,
                evaluation_mask=train_target,
                train_prior=local.train_prior,
                degradation_indicator=indicator_enabled,
            )
            validation_inputs = _evidence_inputs(
                validation_clean_embeddings,
                validation_surface_embeddings,
                local.validation_clean_probabilities,
                local.validation_surface_probabilities,
                validation_source,
                mode="surface",
                evidence_mask=validation_target,
                evaluation_mask=validation_target,
                train_prior=local.train_prior,
                degradation_indicator=indicator_enabled,
            )

            results: dict[str, object] = {}
            validation_probabilities: dict[str, np.ndarray] = {}
            checkpoint_directory = seed_directory / indicator_regime
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            for model_name in ("N0_node_only", "G_repeat"):
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
                    validation_mask=validation_target,
                    validation_metric="accuracy",
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
                graph_records.append(
                    {
                        "seed": seed,
                        "indicator_regime": indicator_regime,
                        "degradation_indicator_enabled": indicator_enabled,
                        "model": model_name,
                        "parameter_count": result.parameter_count,
                        "best_epoch": result.best_epoch,
                        "best_validation_accuracy": result.best_validation_score,
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
                        "indicator_regime": indicator_regime,
                        "degradation_indicator_enabled": indicator_enabled,
                        "training_scenario": "target_surface",
                        "seed": seed,
                        "config_checksum": checksum,
                        "source_checksum": code_checksum,
                        "best_epoch": result.best_epoch,
                        "validation_metric": result.validation_metric,
                        "best_validation_score": result.best_validation_score,
                    },
                    checkpoint_directory / f"{model_name.lower()}_model.pt",
                )

            alpha, alpha_rows = _select_propagation_alpha(
                validation_truth,
                validation_probabilities["N0_node_only"],
                validation_edges["G_repeat"][SINGLE_RELATION],
                labels,
                grid=alpha_grid,
                evaluation_mask=validation_target,
                update_mask=validation_target,
                allowed_neighbor_mask=~validation_inputs.evidence_mask,
                metric="accuracy",
            )
            alpha_records.extend(
                {
                    "seed": seed,
                    "indicator_regime": indicator_regime,
                    "selected": abs(row["alpha"] - alpha) < 1e-12,
                    **row,
                }
                for row in alpha_rows
            )

            for corruption_seed, (
                target_mask,
                component_mask,
                _,
            ) in test_masks.items():
                for base_condition, mode, evidence_mask in (
                    (
                        "clean_targets",
                        "identity",
                        np.zeros(len(test_truth), dtype=bool),
                    ),
                    ("target_surface", "surface", target_mask),
                    ("component_surface", "surface", component_mask),
                ):
                    condition = _condition_name(
                        indicator_regime,
                        base_condition,
                    )
                    inputs = _evidence_inputs(
                        test_clean_embeddings,
                        test_surface_embeddings,
                        local.test_clean_probabilities,
                        local.test_surface_probabilities,
                        test_source,
                        mode=mode,
                        evidence_mask=evidence_mask,
                        evaluation_mask=target_mask,
                        train_prior=local.train_prior,
                        degradation_indicator=indicator_enabled,
                    )
                    probabilities = {
                        model_name: predict_relational_probabilities(
                            result,
                            inputs.features,
                            inputs.local_probabilities,
                            test_edges[model_name],
                            list(RELATION_NAMES),
                            device=device,
                        )
                        for model_name, result in results.items()
                    }
                    probabilities["AVG_repeat"] = _propagate_probabilities(
                        probabilities["N0_node_only"],
                        test_edges["G_repeat"][SINGLE_RELATION],
                        alpha=alpha,
                        update_mask=target_mask,
                        allowed_neighbor_mask=~inputs.evidence_mask,
                    )
                    _record_evaluation(
                        metric_records,
                        prediction_store,
                        test_truth,
                        probabilities,
                        labels,
                        target_mask,
                        seed=seed,
                        training_scenario=f"target_surface_{indicator_regime}",
                        condition=condition,
                        realization=corruption_seed,
                        train_fraction=1.0,
                        checkpoint_condition="target_surface",
                        primary_metric="accuracy",
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
            if prediction_seed != seed:
                continue
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

    primary_family = "R1_no_indicator_primary"
    contrasts = [
        {
            "family": primary_family,
            "contrast": "q_off_G_repeat_minus_N0_target_surface",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "q_off_target_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "q_off_target_surface",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": primary_family,
            "contrast": "indicator_dependence_of_graph_gain",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "q_on_target_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "q_on_target_surface",
                    "model": "N0_node_only",
                },
                {
                    "coefficient": -1,
                    "condition": "q_off_target_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": 1,
                    "condition": "q_off_target_surface",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": primary_family,
            "contrast": "q_off_context_borrowing_target_minus_component",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "q_off_target_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "q_off_target_surface",
                    "model": "N0_node_only",
                },
                {
                    "coefficient": -1,
                    "condition": "q_off_component_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": 1,
                    "condition": "q_off_component_surface",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "R1_reference_q_on_replication",
            "contrast": "q_on_G_repeat_minus_N0_target_surface",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "q_on_target_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "q_on_target_surface",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "R1_reference_average_gain",
            "contrast": "q_off_AVG_repeat_minus_N0_target_surface",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "q_off_target_surface",
                    "model": "AVG_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "q_off_target_surface",
                    "model": "N0_node_only",
                },
            ],
        },
        {
            "family": "R1_reference_graph_vs_average",
            "contrast": "q_off_G_repeat_minus_AVG_repeat_target_surface",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "q_off_target_surface",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "q_off_target_surface",
                    "model": "AVG_repeat",
                },
            ],
        },
        {
            "family": "R1_reference_clean",
            "contrast": "q_off_G_repeat_minus_N0_clean_targets",
            "metric": "accuracy",
            "sesoi": 0.005,
            "terms": [
                {
                    "coefficient": 1,
                    "condition": "q_off_clean_targets",
                    "model": "G_repeat",
                },
                {
                    "coefficient": -1,
                    "condition": "q_off_clean_targets",
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

    summary_records = _summarize_records(
        metric_records,
        keys=("training_scenario", "condition", "model"),
    )
    write_csv(output_directory / "metrics_raw.csv", metric_records)
    write_csv(output_directory / "metrics_summary.csv", summary_records)
    write_csv(
        output_directory / "table_r1_q_indicator_control.csv",
        summary_records,
    )
    write_csv(
        output_directory / "table_r1_q_indicator_contrasts.csv",
        bootstrap_rows,
    )
    write_csv(
        output_directory / "graph_training_diagnostics.csv",
        graph_records,
    )
    write_csv(
        output_directory / "propagation_alpha_selection.csv",
        alpha_records,
    )
    save_json(output_directory / "crossed_bootstrap.json", bootstrap_rows)
    save_json(output_directory / "target_selection.json", target_selection)
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
            "control": "reviewer1_degradation_indicator",
            "indicator_regimes": INDICATOR_REGIMES,
            "surface_only_embedding_cache": str(surface_cache_path),
            "protocol": str(
                project_root.parent.parent
                / "revision"
                / "REVIEWER1_NO_INDICATOR_PROTOCOL.md"
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
        "corruption_seeds": corruption_seeds,
        "trained_relational_heads": trained_heads,
        "bootstrap_contrasts": len(bootstrap_rows),
        "clean_embedding_cache": str(clean_cache_path),
        "surface_only_embedding_cache": str(surface_cache_path),
    }
    save_json(output_directory / "summary.json", summary)
    return summary
