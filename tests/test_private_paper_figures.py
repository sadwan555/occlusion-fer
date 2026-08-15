from pathlib import Path

import pytest

from occlusion_fer.private_paper_figures import build_parser


def test_private_figure_cli_requires_explicit_input_and_output_paths() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "figures"])

    args = parser.parse_args(
        [
            "--private-root",
            "private-results/final-private-test-v2",
            "--output-dir",
            "figures/private-test",
        ]
    )
    assert args.private_root == Path("private-results/final-private-test-v2")
    assert args.output_dir == Path("figures/private-test")


def test_source_tree_has_no_personal_absolute_paths() -> None:
    source_root = Path(__file__).parents[1] / "src" / "occlusion_fer"
    for source_path in source_root.glob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        assert "/Users/" not in source, source_path
        assert "/home/ucla/" not in source, source_path
