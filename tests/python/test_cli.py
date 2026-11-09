from __future__ import annotations

import importlib

import pytest

from fabsim import cli


def _failing_import(error):
    def fake(name, package=None):
        raise error

    return fake


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'tkinter'", name="tkinter"),
        ImportError("libtk8.6.so: cannot open shared object file: No such file or directory"),
        ModuleNotFoundError("No module named 'serial'", name="serial"),
    ],
)
def test_missing_gui_dependency_is_a_message_not_a_traceback(monkeypatch, error):
    monkeypatch.setattr(importlib, "import_module", _failing_import(error))
    with pytest.raises(SystemExit) as stopped:
        cli.main(["stereo"])

    assert "fabsim stereo: missing dependency" in str(stopped.value)
    assert "tkinter" in str(stopped.value)


def test_other_import_errors_are_not_hidden(monkeypatch):
    monkeypatch.setattr(
        importlib,
        "import_module",
        _failing_import(ModuleNotFoundError("x", name="numpy")),
    )

    with pytest.raises(ModuleNotFoundError):
        cli.main(["package"])
