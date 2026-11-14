from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytest
from fixtures_gen import GT_LINES

from pdf_ocrer import worker
from pdf_ocrer.config import DebugConfig, OcrConfig
from pdf_ocrer.ocr_engine import OcrLine
from pdf_ocrer.worker import WorkerTask

_DPI = 200
_FONT = pymupdf.Font("cjk")


class FakeEngine:
    def __init__(self, lines: list[OcrLine]) -> None:
        self.lines = lines
        self.images: list[np.ndarray] = []

    def recognize(self, img_rgb: np.ndarray) -> list[OcrLine]:
        self.images.append(img_rgb.copy())
        return self.lines


class FakeCancel:
    def __init__(self, value: bool) -> None:
        self.value = value
        self.calls = 0

    def is_set(self) -> bool:
        self.calls += 1
        return self.value


class FakeDoc:
    def __init__(self, *, save_error: Exception | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self._save_error = save_error

    def subset_fonts(self) -> None:
        self.calls.append(("subset_fonts",))

    def save(self, output: Path, garbage: int, deflate: bool) -> None:
        self.calls.append(("save", output, garbage, deflate))
        # 先写出半截内容再抛异常，模拟保存中途失败。
        output.write_bytes(b"%PDF-1.7\n% fake worker output\n")
        if self._save_error is not None:
            raise self._save_error

    def close(self) -> None:
        self.calls.append(("close",))


class _FailedProcessReport:
    action = "ocr"


class FakeProcessed:
    """process_pdf 的最小替身：doc 由测试注入。"""

    def __init__(self, doc: FakeDoc) -> None:
        self.reports = [_FailedProcessReport()]
        self.total_pages = 1
        self.ocr_pages = 0
        self.doc = doc


@pytest.fixture(autouse=True)
def reset_worker_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "_engine_factory", _factory_for([]))
    worker.init_worker(OcrConfig(), DebugConfig(), None, None)


def test_process_file_task_missing_source_fails_without_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engines = _install_engine(monkeypatch, [])
    worker.init_worker(OcrConfig(), DebugConfig(), None, None)
    output = tmp_path / "missing_OCR.pdf"

    outcome = worker.process_file_task(
        WorkerTask(
            index=0,
            source=str(tmp_path / "missing.pdf"),
            rel="missing.pdf",
            output=str(output),
            temp_output=str(Path(output).parent / "~test.ocrtmp.pdf"),
        )
    )

    assert outcome.kind == "failed"
    assert "文件不存在" in outcome.note
    assert not output.exists()
    assert engines == []  # 存在性检查先于引擎创建，不加载模型


def test_process_file_task_writes_searchable_output_and_page_events(
    fixtures_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: queue.Queue[tuple[Any, ...]] = queue.Queue()
    engines = _install_engine(monkeypatch, [_gt_ocr_line()])
    worker.init_worker(OcrConfig(), DebugConfig(), None, events)
    output = tmp_path / "scanned_OCR.pdf"

    outcome = worker.process_file_task(
        WorkerTask(
            index=0,
            source=str(fixtures_dir / "scanned.pdf"),
            rel="scanned.pdf",
            output=str(output),
            temp_output=str(Path(output).parent / "~test.ocrtmp.pdf"),
        )
    )

    assert outcome.kind == "ok"
    assert outcome.total_pages == 1
    assert outcome.ocr_pages == 1
    assert outcome.all_existing_text is False
    assert outcome.note == ""
    assert output.exists()
    # 成功路径经原子改名落盘：不留临时文件。
    assert list(tmp_path.glob("~*.ocrtmp.pdf")) == []
    assert _drain_events(events) == [
        ("page_start", 0, "scanned.pdf", 1, 1),
        ("page", 0, "scanned.pdf", 1, 1),
    ]
    assert len(engines) == 1

    doc = pymupdf.open(output)
    try:
        assert "诊断证明书" in doc[0].get_text()
        assert doc[0].search_for("诊断证明书")
    finally:
        doc.close()


def test_process_file_task_existing_text_produces_no_output(
    fixtures_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: queue.Queue[tuple[Any, ...]] = queue.Queue()
    engines = _install_engine(monkeypatch, [_gt_ocr_line()])
    worker.init_worker(OcrConfig(), DebugConfig(), None, events)
    source = fixtures_dir / "native.pdf"
    output = tmp_path / "out" / "native_OCR.pdf"

    outcome = worker.process_file_task(
        WorkerTask(
            index=3,
            source=str(source),
            rel="native.pdf",
            output=str(output),
            temp_output=str(Path(output).parent / "~test.ocrtmp.pdf"),
        )
    )

    assert outcome.kind == "ok"
    # 全部页已含文字层：不产出输出文件。
    assert outcome.total_pages == 1
    assert outcome.ocr_pages == 0
    assert outcome.all_existing_text is True
    assert not output.exists()
    assert _drain_events(events) == [
        ("page_start", 3, "native.pdf", 1, 1),
        ("page", 3, "native.pdf", 1, 1),
    ]
    assert len(engines) == 1
    assert engines[0].images == []


@pytest.mark.parametrize(
    ("filename", "expected_kind", "note_fragment"),
    [
        ("encrypted.pdf", "encrypted", "PDF 需要密码"),
        ("corrupt.pdf", "failed", "FileDataError:"),
    ],
)
def test_process_file_task_error_kinds_keep_existing_output(
    fixtures_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    expected_kind: str,
    note_fragment: str,
) -> None:
    """失败/加密时不产出输出，也绝不删掉已存在的旧输出文件。"""
    events: queue.Queue[tuple[Any, ...]] = queue.Queue()
    _install_engine(monkeypatch, [_gt_ocr_line()])
    worker.init_worker(OcrConfig(), DebugConfig(), None, events)
    output = tmp_path / "out" / f"{filename[:-4]}_OCR.pdf"
    output.parent.mkdir()
    output.write_bytes(b"%PDF-1.7\n% previous complete output")

    outcome = worker.process_file_task(
        WorkerTask(
            index=1,
            source=str(fixtures_dir / filename),
            rel=filename,
            output=str(output),
            temp_output=str(Path(output).parent / "~test.ocrtmp.pdf"),
        )
    )

    assert outcome.kind == expected_kind
    assert outcome.total_pages == 0
    assert outcome.ocr_pages == 0
    assert outcome.all_existing_text is False
    assert note_fragment in outcome.note
    # 已有的完整输出必须原样保留（不能因本次失败被误删）。
    assert output.read_bytes() == b"%PDF-1.7\n% previous complete output"
    assert list(output.parent.glob("~*.ocrtmp.pdf")) == []
    assert _drain_events(events) == []


def test_process_file_task_cancelled_kind_keeps_existing_output(
    fixtures_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: queue.Queue[tuple[Any, ...]] = queue.Queue()
    cancel = FakeCancel(True)
    _install_engine(monkeypatch, [_gt_ocr_line()])
    worker.init_worker(OcrConfig(), DebugConfig(), cancel, events)
    output = tmp_path / "out" / "cancel_OCR.pdf"
    output.parent.mkdir()
    output.write_bytes(b"%PDF-1.7\n% previous complete output")

    outcome = worker.process_file_task(
        WorkerTask(
            index=2,
            source=str(fixtures_dir / "scanned.pdf"),
            rel="scanned.pdf",
            output=str(output),
            temp_output=str(Path(output).parent / "~test.ocrtmp.pdf"),
        )
    )

    assert outcome.kind == "cancelled"
    assert outcome.note == ""
    assert cancel.calls == 1
    assert output.read_bytes() == b"%PDF-1.7\n% previous complete output"
    assert list(output.parent.glob("~*.ocrtmp.pdf")) == []
    assert _drain_events(events) == []


def test_process_file_task_save_failure_cleans_temp_and_final_output(
    fixtures_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """save 中途失败：半截内容只落在临时文件里，且临时文件必须被清理。"""
    events: queue.Queue[tuple[Any, ...]] = queue.Queue()
    _install_engine(monkeypatch, [_gt_ocr_line()])
    worker.init_worker(OcrConfig(), DebugConfig(), None, events)
    output = tmp_path / "scanned_OCR.pdf"
    broken = FakeDoc(save_error=RuntimeError("disk full while saving"))
    monkeypatch.setattr(
        worker,
        "process_pdf",
        lambda *args, **kwargs: FakeProcessed(broken),
    )

    outcome = worker.process_file_task(
        WorkerTask(
            index=0,
            source=str(fixtures_dir / "scanned.pdf"),
            rel="scanned.pdf",
            output=str(output),
            temp_output=str(Path(output).parent / "~test.ocrtmp.pdf"),
        )
    )

    assert outcome.kind == "failed"
    assert "disk full while saving" in outcome.note
    assert not output.exists()
    assert list(tmp_path.glob("~*.ocrtmp.pdf")) == []


# ---------------------------------------------------------------------------
# worker_serve：常驻子进程主循环
# ---------------------------------------------------------------------------


def test_worker_serve_processes_tasks_until_sentinel(tmp_path: Path) -> None:
    task_queue: queue.Queue[object] = queue.Queue()
    outcome_queue: queue.Queue[object] = queue.Queue()
    events_queue: queue.Queue[object] = queue.Queue()
    cancel = threading.Event()
    task_queue.put(
        WorkerTask(
            index=0,
            source=str(tmp_path / "missing.pdf"),
            rel="missing.pdf",
            output=str(tmp_path / "missing_OCR.pdf"),
            temp_output=str(Path(tmp_path / "missing_OCR.pdf").parent / "~test.ocrtmp.pdf"),
        )
    )
    task_queue.put(None)

    thread = threading.Thread(
        target=worker.worker_serve,
        args=(task_queue, outcome_queue, events_queue, cancel, OcrConfig(), DebugConfig()),
    )
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    outcome = outcome_queue.get_nowait()
    assert isinstance(outcome, worker.WorkerOutcome)
    assert outcome.kind == "failed"
    assert "文件不存在" in outcome.note


def test_worker_serve_warmup_reports_error_when_engine_factory_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_factory(cfg: OcrConfig, log: object = None) -> FakeEngine:
        raise RuntimeError("model download failed")

    monkeypatch.setattr(worker, "_engine_factory", broken_factory)
    task_queue: queue.Queue[object] = queue.Queue()
    outcome_queue: queue.Queue[object] = queue.Queue()
    events_queue: queue.Queue[object] = queue.Queue()
    cancel = threading.Event()
    task_queue.put(worker.WARMUP_TASK)
    task_queue.put(None)

    thread = threading.Thread(
        target=worker.worker_serve,
        args=(task_queue, outcome_queue, events_queue, cancel, OcrConfig(), DebugConfig()),
    )
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    kind, note = outcome_queue.get_nowait()
    assert kind == "warmup_error"
    assert "RuntimeError: model download failed" in note


def test_worker_serve_warmup_success_returns_engine_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_queue: queue.Queue[object] = queue.Queue()
    outcome_queue: queue.Queue[object] = queue.Queue()
    events_queue: queue.Queue[object] = queue.Queue()
    cancel = threading.Event()
    task_queue.put(worker.WARMUP_TASK)
    task_queue.put(None)

    thread = threading.Thread(
        target=worker.worker_serve,
        args=(task_queue, outcome_queue, events_queue, cancel, OcrConfig(), DebugConfig()),
    )
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    kind, name = outcome_queue.get_nowait()
    assert kind == "warmup"
    assert "FakeEngine" in name


def test_worker_serve_exits_when_parent_process_dies(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DeadParent:
        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(worker.multiprocessing, "parent_process", lambda: _DeadParent())
    task_queue: queue.Queue[object] = queue.Queue()
    outcome_queue: queue.Queue[object] = queue.Queue()
    events_queue: queue.Queue[object] = queue.Queue()
    cancel = threading.Event()

    thread = threading.Thread(
        target=worker.worker_serve,
        args=(task_queue, outcome_queue, events_queue, cancel, OcrConfig(), DebugConfig()),
    )
    thread.start()
    thread.join(timeout=5)

    # 父进程已死：watchdog 立即退出，不留孤儿进程。
    assert not thread.is_alive()
    assert outcome_queue.empty()


def _install_engine(
    monkeypatch: pytest.MonkeyPatch,
    lines: list[OcrLine],
) -> list[FakeEngine]:
    engines: list[FakeEngine] = []
    monkeypatch.setattr(worker, "_engine_factory", _factory_for(lines, engines))
    return engines


def _factory_for(
    lines: list[OcrLine],
    engines: list[FakeEngine] | None = None,
):
    def factory(cfg: OcrConfig, log: object = None) -> FakeEngine:
        assert isinstance(cfg, OcrConfig)
        assert log is None or callable(log)
        engine = FakeEngine(lines)
        if engines is not None:
            engines.append(engine)
        return engine

    return factory


def _gt_ocr_line() -> OcrLine:
    point, fontsize, text = GT_LINES[0]
    return OcrLine(text, _px_poly(point, fontsize, text), 0.99)


def _px_poly(
    point: tuple[float, float],
    fontsize: float,
    text: str,
) -> tuple[tuple[float, float], ...]:
    baseline_x, baseline_y = point
    top = baseline_y - _FONT.ascender * fontsize
    bottom = baseline_y - _FONT.descender * fontsize
    right = baseline_x + _FONT.text_length(text, fontsize)
    scale = _DPI / 72.0
    return (
        (baseline_x * scale, top * scale),
        (right * scale, top * scale),
        (right * scale, bottom * scale),
        (baseline_x * scale, bottom * scale),
    )


def _drain_events(events: queue.Queue[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    drained: list[tuple[Any, ...]] = []
    while True:
        try:
            drained.append(events.get_nowait())
        except queue.Empty:
            return drained
