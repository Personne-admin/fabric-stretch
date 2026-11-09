from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_stop_command_and_interrupt_notifications(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native firmware checks need g++")
    tests = Path(__file__).resolve().parents[2] / "firmware/tests"
    executable = tmp_path / "firmware-stop"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(tests / "include"),
            str(tests / "stop.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
