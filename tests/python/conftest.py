from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    value = os.environ.get("FABSIM_DATA_DIR")
    if not value or not Path(value).is_dir():
        pytest.skip("FABSIM_DATA_DIR is not set to a measurement data directory")

    return Path(value)
