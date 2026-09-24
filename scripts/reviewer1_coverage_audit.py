from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from mention_graph.data import documents_for_split
from mention_graph.experiments import load_config, load_experiment_documents
from mention_graph.graph import normalize_surface


AUDIT_SALT = "reviewer1-exact-repeat-audit-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    return parser


def _sentence_text(document, mention) -> str:
    for start, end in document.sentence_spans:
        if start <= mention.start < end:
            return " ".join(document.text[start:end].split())
    return ""


def main() -> None:
    arguments = _parser().parse_args()
    project_root = arguments.project_root.resolve()
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    config = load_config(arguments.config.resolve())
    documents = documents_for_split(
        load_experiment_documents(config, project_root),
        "test",
    )

    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    uncovered: list[dict[str, str | int]] = []
    for document in documents:
        forms = Counter(
            normalize_surface(mention.text) for mention in document.mentions
        )
        same_type_forms: dict[str, list[str]] = defaultdict(list)
        for mention in document.mentions:
            same_type_forms[mention.label].append(mention.text)
        for mention in document.mentions:
            normalized = normalize_surface(mention.text)
            covered = forms[normalized] >= 2
            by_type[mention.label]["total"] += 1
            by_type[mention.label]["covered"] += int(covered)
            if covered:
                continue
            digest = hashlib.sha256(
                f"{AUDIT_SALT}\0{mention.mention_id}".encode("utf-8")
            ).hexdigest()
            uncovered.append(
                {
                    "selection_digest": digest,
                    "mention_id": mention.mention_id,
                    "document_id": document.doc_id,
                    "label": mention.label,
                    "surface": mention.text,
                    "normalized_surface": normalized,
                    "start": mention.start,
                    "end": mention.end,
                    "sentence": _sentence_text(document, mention),
                    "same_type_mentions_in_document": " | ".join(
                        same_type_forms[mention.label]
                    ),
                    "document_text": " ".join(document.text.split()),
                }
            )

    coverage_rows = []
    for label in sorted(by_type):
        total = by_type[label]["total"]
        covered = by_type[label]["covered"]
        coverage_rows.append(
            {
                "entity_type": label,
                "covered_mentions": covered,
                "total_mentions": total,
                "coverage_percent": 100.0 * covered / total,
            }
        )
    overall_total = sum(row["total_mentions"] for row in coverage_rows)
    overall_covered = sum(row["covered_mentions"] for row in coverage_rows)
    coverage_rows.append(
        {
            "entity_type": "ALL",
            "covered_mentions": overall_covered,
            "total_mentions": overall_total,
            "coverage_percent": 100.0 * overall_covered / overall_total,
        }
    )

    coverage_path = output_directory / "exact_repeat_coverage_by_type.csv"
    with coverage_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)

    audit_rows = sorted(
        uncovered,
        key=lambda row: str(row["selection_digest"]),
    )[: arguments.sample_size]
    audit_path = output_directory / "uncovered_mentions_audit_sample.tsv"
    with audit_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(audit_rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    metadata = {
        "audit_salt": AUDIT_SALT,
        "sample_size": len(audit_rows),
        "uncovered_population": len(uncovered),
        "covered_population": overall_covered,
        "test_mentions": overall_total,
        "test_documents": len(documents),
        "coverage_file": str(coverage_path),
        "audit_sample_file": str(audit_path),
    }
    (output_directory / "coverage_audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
