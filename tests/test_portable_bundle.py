from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PORTABLE_ROOT = ROOT / "dist_portable" / "RIS_Portable"
PYTHON_EXE = PORTABLE_ROOT / "python" / "python.exe"
RAILWAY_SIM_EXE = PORTABLE_ROOT / "python" / "Scripts" / "railway-sim.exe"


def _portable_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{PORTABLE_ROOT / 'python'};{PORTABLE_ROOT / 'python' / 'Scripts'};{env.get('PATH', '')}"
    # Windows portable smoke tests may run under cp1252 consoles. Force UTF-8
    # so Rich/Typer help and any future engineering symbols cannot crash on
    # encode, while production CLI text is kept ASCII where practical.
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("NO_COLOR", "1")
    env.setdefault("CLICOLOR", "0")
    return env


def test_portable_smoke_script_is_checked_in() -> None:
    script = ROOT / "tools" / "windows-portable" / "Test_Portable_Bundle.ps1"
    launcher = ROOT / "TEST_PORTABLE_BUNDLE.cmd"
    assert script.is_file()
    assert launcher.is_file()
    text = script.read_text(encoding="utf-8")
    assert "railway-sim --help" in text
    assert "study run-grid" in text


@pytest.mark.skipif(not PYTHON_EXE.exists(), reason="portable bundle has not been built")
def test_built_portable_python_imports_runtime_package() -> None:
    env = _portable_env()
    result = subprocess.run(
        [
            str(PYTHON_EXE),
            "-c",
            "import railway_simulator, numpy, pandas, scipy, yaml; print('imports-ok')",
        ],
        cwd=PORTABLE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "imports-ok" in result.stdout


@pytest.mark.skipif(not RAILWAY_SIM_EXE.exists(), reason="portable bundle has not been built")
def test_built_portable_cli_help() -> None:
    env = _portable_env()
    result = subprocess.run(
        [str(RAILWAY_SIM_EXE), "--help"],
        cwd=PORTABLE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Railway impact simulator" in result.stdout
