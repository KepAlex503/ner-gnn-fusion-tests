from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPOSITORY = "https://github.com/lang-uk/ner-uk.git"
# Corpus revision used for every experiment reported in the paper.
REVISION = "7772b45807453854883a8e6d23e4d145014a1a42"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the official NER-UK 2.0 repository."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/external/ner-uk"),
    )
    parser.add_argument(
        "--revision",
        default=REVISION,
        help="git commit of the NER-UK repository to check out",
    )
    arguments = parser.parse_args()
    destination = arguments.destination.resolve()
    split_file = destination / "v2.0" / "data" / "dev-test-split.txt"
    if split_file.exists():
        current = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if current and current != arguments.revision:
            print(
                f"warning: {destination} is at {current}, "
                f"not the expected revision {arguments.revision}"
            )
        print(f"NER-UK 2.0 is already available at {destination}")
        return
    if destination.exists():
        raise SystemExit(
            f"{destination} exists but does not look like NER-UK 2.0; "
            "refusing to overwrite it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPOSITORY, str(destination)], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", arguments.revision],
        check=True,
    )
    if not split_file.exists():
        raise SystemExit("clone completed but the NER-UK 2.0 split file is missing")
    print(f"Downloaded NER-UK 2.0 at {arguments.revision} to {destination}")


if __name__ == "__main__":
    main()

