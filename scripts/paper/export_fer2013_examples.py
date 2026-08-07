#!/usr/bin/env python3
"""Export deterministic FER2013 class examples for paper use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence
import sys

from occlusion_fer.paper_figures import export_fer2013_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export deterministic Training/PublicTest FER2013 class examples. "
            "PrivateTest is never available."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=("Training", "PublicTest"),
        default="Training",
    )
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    try:
        provenance = export_fer2013_examples(
            args.data_path,
            args.output_dir,
            split=args.split,
            samples_per_class=args.samples_per_class,
            creation_command=command,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(provenance, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
