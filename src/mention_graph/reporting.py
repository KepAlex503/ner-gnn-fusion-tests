from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


MAIN_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "micro_f1",
)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def write_main_tables(
    output_directory: Path,
    run_records: list[dict[str, object]],
    *,
    labels: list[str],
) -> None:
    raw_rows: list[dict[str, object]] = []
    for record in run_records:
        metrics = record["metrics"]
        row: dict[str, object] = {
            "seed": record["seed"],
            "model": record["model"],
        }
        row.update({metric: metrics[metric] for metric in MAIN_METRICS})
        raw_rows.append(row)
    write_csv(output_directory / "metrics_by_seed.csv", raw_rows)

    calibration_rows: list[dict[str, object]] = []
    for record in run_records:
        metrics = record["metrics"]
        calibration_rows.append(
            {
                "seed": record["seed"],
                "model": record["model"],
                "negative_log_likelihood": metrics[
                    "negative_log_likelihood"
                ],
                "brier_score": metrics["brier_score"],
                "ece_10_bins": metrics["ece_10_bins"],
            }
        )
    write_csv(
        output_directory / "calibration_by_seed.csv",
        calibration_rows,
    )

    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in run_records:
        by_model[str(record["model"])].append(record)
    summary_rows: list[dict[str, object]] = []
    for model, records in sorted(by_model.items()):
        row: dict[str, object] = {"model": model, "runs": len(records)}
        for metric in MAIN_METRICS:
            mean, standard_deviation = _mean_std(
                float(record["metrics"][metric]) for record in records
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = standard_deviation
            row[f"{metric}_formatted"] = (
                f"{mean:.4f} ± {standard_deviation:.4f}"
            )
        summary_rows.append(row)
    write_csv(output_directory / "metrics_summary.csv", summary_rows)

    e1_models = ("local", "repeat_consistency", "graph_all")
    write_csv(
        output_directory / "table_e1.csv",
        [row for row in summary_rows if row["model"] in e1_models],
    )

    local_macro = next(
        (
            float(row["macro_f1_mean"])
            for row in summary_rows
            if row["model"] == "local"
        ),
        float("nan"),
    )
    e2_rows: list[dict[str, object]] = []
    for row in summary_rows:
        if not str(row["model"]).startswith("graph_"):
            continue
        e2_row = dict(row)
        delta = float(row["macro_f1_mean"]) - local_macro
        e2_row["delta_macro_f1_vs_local"] = delta
        e2_row["relative_delta_vs_local"] = (
            delta / local_macro if local_macro else float("nan")
        )
        e2_rows.append(e2_row)
    write_csv(output_directory / "table_e2.csv", e2_rows)

    e3_rows: list[dict[str, object]] = []
    for model in e1_models:
        records = by_model.get(model, [])
        if not records:
            continue
        for label in labels:
            f1_mean, f1_std = _mean_std(
                float(record["metrics"]["per_class"][label]["f1"])
                for record in records
            )
            precision_mean, precision_std = _mean_std(
                float(record["metrics"]["per_class"][label]["precision"])
                for record in records
            )
            recall_mean, recall_std = _mean_std(
                float(record["metrics"]["per_class"][label]["recall"])
                for record in records
            )
            support = int(records[0]["metrics"]["per_class"][label]["support"])
            e3_rows.append(
                {
                    "label": label,
                    "model": model,
                    "support": support,
                    "precision_mean": precision_mean,
                    "precision_std": precision_std,
                    "recall_mean": recall_mean,
                    "recall_std": recall_std,
                    "f1_mean": f1_mean,
                    "f1_std": f1_std,
                    "f1_formatted": f"{f1_mean:.4f} ± {f1_std:.4f}",
                }
            )
    write_csv(output_directory / "table_e3.csv", e3_rows)

    confusion_directory = output_directory / "confusion_matrices"
    for model in e1_models:
        records = by_model.get(model, [])
        if not records:
            continue
        aggregate = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for record in records:
            matrix = np.asarray(
                record["metrics"]["confusion_matrix"],
                dtype=np.int64,
            )
            aggregate += matrix
            rows = []
            for row_index, label in enumerate(labels):
                row: dict[str, object] = {"gold_label": label}
                row.update(
                    {
                        f"pred_{predicted_label}": int(
                            matrix[row_index, column_index]
                        )
                        for column_index, predicted_label in enumerate(labels)
                    }
                )
                rows.append(row)
            write_csv(
                confusion_directory
                / f"{model}_seed-{record['seed']}.csv",
                rows,
            )
        aggregate_rows = []
        for row_index, label in enumerate(labels):
            row = {"gold_label": label}
            row.update(
                {
                    f"pred_{predicted_label}": int(
                        aggregate[row_index, column_index]
                    )
                    for column_index, predicted_label in enumerate(labels)
                }
            )
            aggregate_rows.append(row)
        write_csv(
            confusion_directory / f"{model}_aggregate.csv",
            aggregate_rows,
        )


def write_dataset_table(
    output_directory: Path,
    statistics: dict[str, object],
) -> None:
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        values = statistics["splits"][split]
        rows.append(
            {
                "split": split,
                "documents": values["documents"],
                "sentences": values["sentences"],
                "mentions": values["mentions"],
                "classes_present": values["classes_present"],
            }
        )
    write_csv(output_directory / "table_dataset.csv", rows)


def write_repetition_table(
    output_directory: Path,
    records: list[dict[str, object]],
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["bucket"]), str(record["model"]))].append(record)
    rows: list[dict[str, object]] = []
    order = {"1": 0, "2": 1, "3+": 2}
    for (bucket, model), bucket_records in sorted(
        grouped.items(),
        key=lambda item: (order.get(item[0][0], 99), item[0][1]),
    ):
        mean, standard_deviation = _mean_std(
            float(record["metrics"]["macro_f1"]) for record in bucket_records
        )
        rows.append(
            {
                "frequency_in_document": bucket,
                "model": model,
                "mentions": bucket_records[0]["metrics"]["support"],
                "macro_f1_mean": mean,
                "macro_f1_std": standard_deviation,
                "macro_f1_formatted": f"{mean:.4f} ± {standard_deviation:.4f}",
            }
        )
    local_lookup = {
        row["frequency_in_document"]: float(row["macro_f1_mean"])
        for row in rows
        if row["model"] == "local"
    }
    for row in rows:
        if row["model"] == "graph_all":
            row["absolute_change_vs_local"] = (
                float(row["macro_f1_mean"])
                - local_lookup[str(row["frequency_in_document"])]
            )
    write_csv(output_directory / "table_e4.csv", rows)


def write_robustness_table(
    output_directory: Path,
    records: list[dict[str, object]],
) -> None:
    if not records:
        return
    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["noise_kind"]),
                float(record["noise_level"]),
                str(record["model"]),
            )
        ].append(record)
    rows: list[dict[str, object]] = []
    for key, condition_records in sorted(grouped.items()):
        kind, level, model = key
        mean, standard_deviation = _mean_std(
            float(record["metrics"]["macro_f1"]) for record in condition_records
        )
        clean_mean, _ = _mean_std(
            float(record["clean_macro_f1"]) for record in condition_records
        )
        rows.append(
            {
                "noise_kind": kind,
                "noise_level": level,
                "model": model,
                "macro_f1_mean": mean,
                "macro_f1_std": standard_deviation,
                "clean_macro_f1_mean": clean_mean,
                "drop_macro_f1": clean_mean - mean,
            }
        )
    write_csv(output_directory / "table_e5.csv", rows)
