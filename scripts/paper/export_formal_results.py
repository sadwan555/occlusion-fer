#!/usr/bin/env python3
"""Export cross-seed paper tables and figures from formal result artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from occlusion_fer.paper_results import FORMAL_SEEDS, export_formal_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate exactly three clean-only and three mixed PublicTest "
            "evaluation directories into paper-ready CSV and PNG artifacts."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--clean-run",
        action="append",
        required=True,
        metavar="SEED=EVALUATION_DIR",
    )
    parser.add_argument(
        "--mixed-run",
        action="append",
        required=True,
        metavar="SEED=EVALUATION_DIR",
    )
    parser.add_argument(
        "--clean-training-run",
        action="append",
        metavar="SEED=TRAINING_DIR",
    )
    parser.add_argument(
        "--mixed-training-run",
        action="append",
        metavar="SEED=TRAINING_DIR",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _seed_paths(values: Sequence[str], option: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError(f"{option} seed must be an integer") from exc
        if not separator or not path_text:
            raise ValueError(f"{option} must use SEED=PATH")
        if seed in result:
            raise ValueError(f"{option} contains duplicate seed {seed}")
        result[seed] = Path(path_text)
    if set(result) != set(FORMAL_SEEDS):
        raise ValueError(f"{option} must contain seeds 42, 123, 2026")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    try:
        evaluation_runs = {
            "clean-only": _seed_paths(args.clean_run, "--clean-run"),
            "mixed": _seed_paths(args.mixed_run, "--mixed-run"),
        }
        has_clean_training = args.clean_training_run is not None
        has_mixed_training = args.mixed_training_run is not None
        if has_clean_training != has_mixed_training:
            raise ValueError(
                "clean and mixed training-run options must be supplied together"
            )
        training_runs = None
        if has_clean_training:
            training_runs = {
                "clean-only": _seed_paths(
                    args.clean_training_run, "--clean-training-run"
                ),
                "mixed": _seed_paths(
                    args.mixed_training_run, "--mixed-training-run"
                ),
            }
        provenance = export_formal_results(
            evaluation_runs,
            args.output_dir,
            training_runs=training_runs,
            creation_command=command,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(provenance, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
