from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import corpus_statistics
from .diagnostic_pipeline import run_diagnostic_experiment
from .experiments import (
    load_config,
    load_experiment_documents,
    run_experiment,
)
from .followup import run_followup_experiment
from .reviewer1_control import run_reviewer1_control
from .reviewer2_morphology import run_reviewer2_morphology


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mention-graph",
        description=(
            "Experiments for graph-based refinement of known Ukrainian NER spans."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run E1-E5")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--project-root", type=Path, default=Path.cwd())

    followup_parser = subparsers.add_parser(
        "run-followup",
        help="run the relation-aware E6-E10 follow-up protocol",
    )
    followup_parser.add_argument("--config", type=Path, required=True)
    followup_parser.add_argument("--output", type=Path)
    followup_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    diagnostic_parser = subparsers.add_parser(
        "run-diagnostics",
        help="run the oracle, recovery, low-resource, and ambiguity protocol",
    )
    diagnostic_parser.add_argument("--config", type=Path, required=True)
    diagnostic_parser.add_argument("--output", type=Path)
    diagnostic_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    reviewer1_parser = subparsers.add_parser(
        "run-reviewer1-control",
        help="run the Reviewer 1 q_i on/off matched control",
    )
    reviewer1_parser.add_argument("--config", type=Path, required=True)
    reviewer1_parser.add_argument("--output", type=Path)
    reviewer1_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    reviewer2_parser = subparsers.add_parser(
        "run-reviewer2-morphology",
        help="run the Reviewer 2 exact-repeat versus lemma-edge control",
    )
    reviewer2_parser.add_argument("--config", type=Path, required=True)
    reviewer2_parser.add_argument("--output", type=Path)
    reviewer2_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    inspect_parser = subparsers.add_parser(
        "inspect-data",
        help="validate a configured dataset and print its statistics",
    )
    inspect_parser.add_argument("--config", type=Path, required=True)
    inspect_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    config_path = arguments.config.resolve()
    project_root = arguments.project_root.resolve()
    config = load_config(config_path)
    if arguments.command == "inspect-data":
        documents = load_experiment_documents(config, project_root)
        print(
            json.dumps(
                corpus_statistics(documents),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output = arguments.output
    if output is None:
        output = project_root / "artifacts" / str(config["name"])
    elif not output.is_absolute():
        output = project_root / output
    if arguments.command == "run-followup":
        runner = run_followup_experiment
    elif arguments.command == "run-diagnostics":
        runner = run_diagnostic_experiment
    elif arguments.command == "run-reviewer1-control":
        runner = run_reviewer1_control
    elif arguments.command == "run-reviewer2-morphology":
        runner = run_reviewer2_morphology
    else:
        runner = run_experiment
    summary = runner(
        config,
        project_root=project_root,
        output_directory=output.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
