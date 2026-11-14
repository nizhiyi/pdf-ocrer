"""需要真实 OCR 模型的集成冒烟测试。

本文件名不带 ``test_`` 前缀，默认 ``pytest`` 收集完全跳过（单元测试输出
里不会出现 "deselected"）。运行方式：

    pytest -o python_files="integration_smoke.py" tests/integration_smoke.py
"""

from __future__ import annotations

import numpy as np
import pymupdf

from pdf_ocrer.config import AppConfig, DebugConfig, OcrConfig, OutputConfig
from pdf_ocrer.rapidocr_engine import RapidOcrEngine


def make_cfg() -> AppConfig:
    return AppConfig(ocr=OcrConfig(), output=OutputConfig(), debug=DebugConfig())


def _normalize(text: str) -> str:
    """OCR 对标点/空格的识别存在固有差异（半角↔全角、丢空格）：归一化后比较。"""
    return text.replace("：", ":").replace(" ", "")


def test_real_rapidocr_engine_recognizes_scanned_fixture(fixtures_dir) -> None:
    doc = pymupdf.open(fixtures_dir / "scanned.pdf")
    pix = doc[0].get_pixmap(dpi=200, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    lines = RapidOcrEngine(OcrConfig()).recognize(img)

    assert "诊断证明书" in [line.text for line in lines]


def test_real_process_pdf_adds_searchable_text_layer(fixtures_dir, tmp_path) -> None:
    from fixtures_gen import GT_LINES

    from pdf_ocrer.pdf_processor import process_pdf

    cfg = AppConfig(
        ocr=OcrConfig(),
        output=OutputConfig(),
        debug=DebugConfig(),
    )
    engine = RapidOcrEngine(cfg.ocr)

    scanned = process_pdf(fixtures_dir / "scanned.pdf", cfg, engine)
    scanned_out = tmp_path / "scanned_ocr.pdf"
    scanned.doc.subset_fonts()
    scanned.doc.save(scanned_out, garbage=3, deflate=True)
    scanned_page = pymupdf.open(scanned_out)[0]
    scanned_text = _normalize(scanned_page.get_text())

    for _, _, expected in GT_LINES:
        assert _normalize(expected) in scanned_text
    scanned_rects = scanned_page.search_for("诊断证明书")
    assert scanned_rects
    assert abs(scanned_rects[0].x0 - 72) < 20

    rotated = process_pdf(fixtures_dir / "rotated.pdf", cfg, engine)
    rotated_out = tmp_path / "rotated_ocr.pdf"
    rotated.doc.subset_fonts()
    rotated.doc.save(rotated_out, garbage=3, deflate=True)
    rotated_page = pymupdf.open(rotated_out)[0]

    assert rotated_page.search_for("诊断证明书")


def test_real_worker_session_spawn_smoke(fixtures_dir, tmp_path) -> None:
    """真 spawn 子进程跑常驻 worker：native 跳过、corrupt 失败、无临时文件。"""
    from pdf_ocrer.pipeline import FileStatus
    from pdf_ocrer.runner import WorkerSession

    (tmp_path / "native.pdf").write_bytes((fixtures_dir / "native.pdf").read_bytes())
    (tmp_path / "corrupt.pdf").write_bytes((fixtures_dir / "corrupt.pdf").read_bytes())
    session = WorkerSession()  # 不注入：真 spawn 子进程 + 真 worker_serve
    try:
        summary = session.run_batch([tmp_path], make_cfg(), None)
    finally:
        session.shutdown()

    assert summary.cancelled is False
    assert [result.rel for result in summary.results] == ["corrupt.pdf", "native.pdf"]
    assert [result.status for result in summary.results] == [
        FileStatus.FAILED,
        FileStatus.SUCCESS_EXISTING_TEXT,
    ]
    assert "FileDataError" in summary.results[0].note
    assert summary.results[1].output is None
    assert not list(tmp_path.rglob("~*.ocrtmp.pdf"))
