from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "FacebookAI/xlm-roberta-base"
# Hugging Face commit used for every experiment reported in the paper.
REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the files needed for the frozen XLM-R encoder."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/models/xlm-roberta-base"),
    )
    arguments = parser.parse_args()
    destination = arguments.destination.resolve()
    if all((destination / filename).exists() for filename in REQUIRED_FILES):
        print(f"XLM-R is already available at {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=list(REQUIRED_FILES),
    )
    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (destination / filename).exists()
    ]
    if missing:
        raise SystemExit(f"download is incomplete; missing: {missing}")
    print(f"Downloaded {MODEL_ID}@{REVISION} to {destination}")


if __name__ == "__main__":
    main()

