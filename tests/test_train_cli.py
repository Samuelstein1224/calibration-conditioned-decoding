"""CLI smoke tests for scripts/train.py model selection."""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN = REPO_ROOT / "scripts" / "train.py"


def _run(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(TRAIN), *args],
        capture_output=True, text=True, env=env,
    )


def test_help_exposes_model_flag():
    res = _run("--help")
    assert res.returncode == 0
    assert "--model" in res.stdout
    assert "{cnn,film}" in res.stdout


def test_invalid_model_choice_rejected():
    # argparse rejects unknown choices before any training happens
    res = _run("some_dir", "--model", "transformer")
    assert res.returncode != 0
    assert "invalid choice" in res.stderr
