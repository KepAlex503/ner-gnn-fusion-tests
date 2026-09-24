from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mention_graph.models import RelationalGatedClassifier
from mention_graph.relational import RELATION_NAMES


def _count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _influencing(rows, column: str, indicator_varies: bool) -> int:
    active = {"yes", "q_varies"} if indicator_varies else {"yes"}
    return sum(
        int(row["parameters"])
        for row in rows
        if row["parameter_group"] not in {"nominal total"}
        and row[column] in active
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    model = RelationalGatedClassifier(
        input_dimension=785,
        hidden_dimension=128,
        class_count=13,
        relation_names=list(RELATION_NAMES),
        layers=1,
        dropout=0.2,
        layer_normalization=True,
    )
    layer = model.layers[0]
    # The degradation indicator q_i is the last input feature column.
    indicator_weights = model.input_projection.weight[:, -1].numel()
    rows = [
        {
            "parameter_group": "input projection W_0, b_0 excluding q_i column",
            "parameters": _count(model.input_projection) - indicator_weights,
            "influences_N0": "yes",
            "influences_G_repeat": "yes",
        },
        {
            "parameter_group": "input projection weights on q_i column",
            "parameters": indicator_weights,
            "influences_N0": "q_varies",
            "influences_G_repeat": "q_varies",
        },
        {
            "parameter_group": "self update W_s, b_s",
            "parameters": _count(layer.self_projection),
            "influences_N0": "yes",
            "influences_G_repeat": "yes",
        },
        {
            "parameter_group": "active neighbor projection W_n",
            "parameters": _count(layer.neighbor_projections["sent"]),
            "influences_N0": "no",
            "influences_G_repeat": "yes",
        },
        {
            "parameter_group": "active neighbor gate w_g, b_g",
            "parameters": _count(layer.gates["sent"]),
            "influences_N0": "no",
            "influences_G_repeat": "yes",
        },
        {
            "parameter_group": "three reserved inactive relation branches",
            "parameters": sum(
                _count(layer.neighbor_projections[name])
                + _count(layer.gates[name])
                for name in RELATION_NAMES
                if name != "sent"
            ),
            "influences_N0": "no",
            "influences_G_repeat": "no",
        },
        {
            "parameter_group": "LayerNorm",
            "parameters": _count(layer.normalization),
            "influences_N0": "yes",
            "influences_G_repeat": "yes",
        },
        {
            "parameter_group": "correction projection W_c, b_c'",
            "parameters": _count(model.correction_projection),
            "influences_N0": "yes",
            "influences_G_repeat": "yes",
        },
        {
            "parameter_group": "correction gate w_c, b_c",
            "parameters": _count(model.correction_gate),
            "influences_N0": "yes",
            "influences_G_repeat": "yes",
        },
        {
            "parameter_group": "local-logit scale s",
            "parameters": model.local_logit_scale.numel(),
            "influences_N0": "yes",
            "influences_G_repeat": "yes",
        },
    ]
    nominal_total = sum(int(row["parameters"]) for row in rows)
    model_total = _count(model)
    if nominal_total != model_total:
        raise RuntimeError(
            f"breakdown {nominal_total} differs from model total {model_total}"
        )
    rows.extend(
        [
            {
                "parameter_group": "nominal total",
                "parameters": nominal_total,
                "influences_N0": "185747",
                "influences_G_repeat": "185747",
            },
            {
                "parameter_group": "prediction-influencing total, q_i varies",
                "parameters": "",
                "influences_N0": _influencing(rows, "influences_N0", True),
                "influences_G_repeat": _influencing(
                    rows, "influences_G_repeat", True
                ),
            },
            {
                "parameter_group": "prediction-influencing total, q_i constant",
                "parameters": "",
                "influences_N0": _influencing(rows, "influences_N0", False),
                "influences_G_repeat": _influencing(
                    rows, "influences_G_repeat", False
                ),
            },
        ]
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"nominal={nominal_total}; "
        f"N0_effective={rows[-2]['influences_N0']}/{rows[-1]['influences_N0']}; "
        f"G_repeat_effective="
        f"{rows[-2]['influences_G_repeat']}/{rows[-1]['influences_G_repeat']}"
    )


if __name__ == "__main__":
    main()
