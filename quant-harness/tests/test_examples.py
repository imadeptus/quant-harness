"""Smoke test: the quickstart example must keep running end to end.

Examples are documentation that executes — if this breaks, the README's promised
'runs with no data download' is a lie. Run as a subprocess so the __main__ path,
the public-API imports, and the internal PASS/KILL asserts are all exercised.
"""
import subprocess
import sys
from pathlib import Path


def test_quickstart_example_runs():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "examples" / "quickstart.py")],
        capture_output=True, text=True, cwd=root)
    assert proc.returncode == 0, f"example failed:\n{proc.stderr}"
    assert "verdict=KILL" in proc.stdout, proc.stdout
    assert "verdict=PASS" in proc.stdout, proc.stdout
