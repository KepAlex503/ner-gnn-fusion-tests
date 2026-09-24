from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


CATEGORIES = {
    "genuine_singleton_or_no_identifiable_coreferent",
    "inflectional_variant",
    "abbreviation_or_name_shortening",
    "orthographic_variant",
    "other_lexical_variant",
    "uncertain",
}
NON_MISSED = {
    "genuine_singleton_or_no_identifiable_coreferent",
    "uncertain",
}


def _rows(path: Path, *, delimiter: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def _wilson(successes: int, total: int, z: float = 1.95996398454) -> list[float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - radius, center + radius]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    sample = _rows(arguments.sample, delimiter="\t")
    annotations = _rows(arguments.annotations, delimiter="\t")
    sample_ids = [row["mention_id"] for row in sample]
    annotation_ids = [row["mention_id"] for row in annotations]
    if sample_ids != annotation_ids:
        raise ValueError("annotation rows do not match the fixed audit sample")
    unknown = {row["category"] for row in annotations} - CATEGORIES
    if unknown:
        raise ValueError(f"unknown audit categories: {sorted(unknown)}")

    counts = Counter(row["category"] for row in annotations)
    missed = sum(
        count for category, count in counts.items() if category not in NON_MISSED
    )
    result = {
        "sample_size": len(annotations),
        "category_counts": {
            category: counts[category] for category in sorted(CATEGORIES)
        },
        "manually_identifiable_missed_coreferents": missed,
        "manually_identifiable_missed_coreferent_fraction": (
            missed / len(annotations)
        ),
        "wilson_confidence_interval_95": _wilson(
            missed,
            len(annotations),
        ),
        "interpretation": (
            "single-auditor descriptive estimate; not gold-standard "
            "same-entity link recall"
        ),
    }
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
