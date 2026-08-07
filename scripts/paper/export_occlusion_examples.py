#!/usr/bin/env python3
"""Export the clean plus nine formal v2 occlusion examples."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from occlusion_fer.paper_figures import export_occlusion_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export one Training/PublicTest sample under the exact "
            "occlusion-v2-224 protocol. PrivateTest is never available."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument(
        "--split",
        required=True,
        choices=("Training", "PublicTest"),
    )
    parser.add_argument("--sample-id", required=True, type=int)
    parser.add_argument("--training-mean", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Required for PublicTest; canonical publictest_manifest_v2.csv.",
    )
    parser.add_argument(
        "--manifest-sidecar",
        type=Path,
        help="Defaults to the manifest path with a .json suffix.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    try:
        provenance = export_occlusion_examples(
            args.data_path,
            args.output_dir,
            split=args.split,
            sample_id=args.sample_id,
            training_mean_path=args.training_mean,
            manifest_path=args.manifest,
            manifest_sidecar_path=args.manifest_sidecar,
            creation_command=command,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(provenance, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
