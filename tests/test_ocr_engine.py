from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pdf_ocrer.config import OcrConfig
from pdf_ocrer.ocr_engine import create_engine


def test_module_import_is_lazy_about_engine_libs() -> None:
    code = (
        "import pdf_ocrer.ocr_engine, sys; "
        "assert 'rapidocr' not in sys.modules; "
        "assert 'paddleocr' not in sys.modules"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_create_engine_builds_rapidocr_with_cfg_and_log(monkeypatch) -> None:
    logs: list[str] = []

    class FakeRapidOcrEngine:
        def __init__(self, cfg: OcrConfig, log: object = None) -> None:
            self.cfg = cfg
            self.log = log

    fake_module = ModuleType("pdf_ocrer.rapidocr_engine")
    fake_module.RapidOcrEngine = FakeRapidOcrEngine
    monkeypatch.setitem(sys.modules, "pdf_ocrer.rapidocr_engine", fake_module)

    cfg = OcrConfig()
    engine = create_engine(cfg, logs.append)

    assert isinstance(engine, FakeRapidOcrEngine)
    assert engine.cfg is cfg
    assert engine.log == logs.append


def test_create_engine_missing_dependency_in_frozen_build_raises(monkeypatch) -> None:
    from pdf_ocrer import deps

    monkeypatch.setattr(deps, "_all_modules_available", lambda modules: False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    with pytest.raises(Exception, match="安装版"):
        create_engine(OcrConfig())
