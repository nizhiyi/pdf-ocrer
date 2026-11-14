from __future__ import annotations

import importlib.util
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pdf_ocrer.config import ConfigError, OcrConfig
from pdf_ocrer.rapidocr_engine import (
    RapidOcrEngine,
    _rapidocr_params,
    lines_from_rapidocr,
)


def test_lines_from_rapidocr_filters_low_confidence_and_blank_text() -> None:
    poly = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.int16)
    output = SimpleNamespace(
        boxes=np.array([poly, poly, poly], dtype=np.int16),
        txts=("保留", "低分", "  "),
        scores=(0.91, 0.49, 0.99),
    )

    lines = lines_from_rapidocr(output, 0.5)

    assert [line.text for line in lines] == ["保留"]


def test_rapidocr_engine_requires_onnxruntime_even_when_rapidocr_is_loaded(monkeypatch) -> None:
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "onnxruntime":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=object))
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(ConfigError, match=r"pip install pdf-ocrer\[rapidocr\]"):
        RapidOcrEngine(OcrConfig())


def test_pyproject_ships_rapidocr_in_main_dependencies() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    deps = pyproject["project"]["dependencies"]
    assert "rapidocr>=3.9" in deps
    assert "onnxruntime>=1.19" in deps


def test_pyproject_defines_gpu_extras() -> None:
    extras = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]

    assert extras["rapidocr-gpu-cuda"] == ["onnxruntime-gpu>=1.19"]
    assert extras["rapidocr-gpu-dml"] == ["onnxruntime-directml>=1.19"]


def test_rapidocr_params_maps_non_small_model_type_as_enum() -> None:
    from rapidocr.utils.typings import ModelType, OCRVersion

    params = _rapidocr_params(replace(OcrConfig(), ocr_version="PP-OCRv4", model_type="server"))

    assert params["Det.model_type"] == ModelType("server")
    assert params["Rec.model_type"] == ModelType("server")
    assert params["Det.ocr_version"] == OCRVersion("PP-OCRv4")
    assert params["Rec.ocr_version"] == OCRVersion("PP-OCRv4")


def test_rapidocr_params_maps_v6_medium_as_enum() -> None:
    from rapidocr.utils.typings import ModelType, OCRVersion

    params = _rapidocr_params(replace(OcrConfig(), ocr_version="PP-OCRv6", model_type="medium"))

    assert params["Det.model_type"] == ModelType("medium")
    assert params["Rec.model_type"] == ModelType("medium")
    assert params["Det.ocr_version"] == OCRVersion("PP-OCRv6")
    assert params["Rec.ocr_version"] == OCRVersion("PP-OCRv6")


