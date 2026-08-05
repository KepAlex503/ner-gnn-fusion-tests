from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPOSITORY = "https://github.com/lang-uk/ner-uk.git"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the official NER-UK 2.0 repository."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/external/ner-uk"),
    )
    arguments = parser.parse_args()
    destination = arguments.destination.resolve()
    split_file = destination / "v2.0" / "data" / "dev-test-split.txt"
    if split_file.exists():
        print(f"NER-UK 2.0 is already available at {destination}")
        return
    if destination.exists():
        raise SystemExit(
            f"{destination} exists but does not look like NER-UK 2.0; "
            "refusing to overwrite it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPOSITORY, str(destination)],
        check=True,
    )
    if not split_file.exists():
        raise SystemExit("clone completed but the NER-UK 2.0 split file is missing")
    print(f"Downloaded NER-UK 2.0 to {destination}")


if __name__ == "__main__":
    main()

