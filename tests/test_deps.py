from __future__ import annotations

import subprocess
import sys

import pytest

from pdf_ocrer.config import ConfigError
from pdf_ocrer.deps import ensure_engine_deps


def test_engine_with_all_deps_present_does_not_run_pip(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    ensure_engine_deps("rapidocr")

    assert calls == []


def test_missing_rapidocr_runs_expected_pip_command(monkeypatch) -> None:
    state = {"installed": False}
    monkeypatch.setattr(
        "pdf_ocrer.deps._all_modules_available",
        lambda modules: state["installed"],
    )
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        state["installed"] = True
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    ensure_engine_deps("rapidocr")

    assert commands == [[sys.executable, "-m", "pip", "install", "rapidocr>=3.9", "onnxruntime>=1.19"]]


def test_pip_failure_raises_config_error_with_manual_command(monkeypatch) -> None:
    monkeypatch.setattr("pdf_ocrer.deps._all_modules_available", lambda modules: False)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ERROR: no matching distribution")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ConfigError, match=r"pip install rapidocr>=3\.9"):
        ensure_engine_deps("rapidocr")


def test_pip_success_but_module_still_missing_raises(monkeypatch) -> None:
    monkeypatch.setattr("pdf_ocrer.deps._all_modules_available", lambda modules: False)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ConfigError):
        ensure_engine_deps("rapidocr")


def test_frozen_build_missing_deps_raises_without_pip(monkeypatch) -> None:
    monkeypatch.setattr("pdf_ocrer.deps._all_modules_available", lambda modules: False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    with pytest.raises(ConfigError, match="安装版"):
        ensure_engine_deps("rapidocr")

    assert calls == []
