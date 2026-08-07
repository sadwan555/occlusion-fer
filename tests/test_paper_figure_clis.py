from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_paper_figure_scripts_expose_help() -> None:
    scripts = {
        "scripts/paper/export_fer2013_examples.py": "--data-path",
        "scripts/paper/export_occlusion_examples.py": "--data-path",
        "scripts/paper/export_formal_results.py": "--clean-run",
    }
    for relative_path, required_option in scripts.items():
        result = subprocess.run(
            [sys.executable, relative_path, "--help"],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert required_option in result.stdout
        assert "--output-dir" in result.stdout
