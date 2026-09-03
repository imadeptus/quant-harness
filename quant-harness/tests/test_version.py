"""harness.__version__ is the single version string every report and API
response quotes; it must equal [project].version in pyproject.toml."""
from __future__ import annotations

import tomllib
from pathlib import Path

import harness


def test_package_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    assert harness.__version__ == project["version"]
