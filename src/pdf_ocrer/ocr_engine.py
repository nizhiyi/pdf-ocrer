"""OCR engine construction.

The only supported engine is RapidOCR (ONNX Runtime). ``create_engine`` is kept
as the single construction point shared by the GUI worker, CLI, and tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pdf_ocrer.config import OcrConfig
from pdf_ocrer.deps import ensure_engine_deps


@dataclass(frozen=True)
class OcrLine:
    text: str
    poly: tuple[tuple[float, float], ...]
    score: float


class OcrEngineProtocol(Protocol):
    def recognize(self, img_rgb: np.ndarray) -> list[OcrLine]: ...


def create_engine(cfg: OcrConfig, log: Callable[[str], None] | None = None) -> OcrEngineProtocol:
    # 缺依赖时自动 pip 安装（源码/pip 安装；冻结版内会给出明确错误）。
    ensure_engine_deps("rapidocr", log)

    from pdf_ocrer.rapidocr_engine import RapidOcrEngine

    return RapidOcrEngine(cfg, log)
