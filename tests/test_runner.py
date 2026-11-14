from __future__ import annotations

import queue
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from pdf_ocrer.config import AppConfig, DebugConfig, InputConfig, OcrConfig, OutputConfig
from pdf_ocrer.pipeline import FileResult, FileStatus
from pdf_ocrer.runner import WorkerSession
from pdf_ocrer.worker import WorkerOutcome, WorkerTask

_WARMUP_TASK = "warmup"


def make_cfg(**overrides: object) -> AppConfig:
    values = {
        "ocr": OcrConfig(),
        "output": OutputConfig(),
        "debug": DebugConfig(),
    }
    values.update(overrides)
    return AppConfig(**values)


# ---------------------------------------------------------------------------
# 注入设施：用进程内线程替身代替 spawn 子进程，队列语义与真实现一致。
# ---------------------------------------------------------------------------


class _FakeProcess:
    """threading.Thread 的进程句柄替身（is_alive/join/terminate）。

    terminate/kill 只能以协作方式通知线程退出（Python 线程不可强杀），
    因此用 stop_event 模拟真进程被终止后 serve 循环退出的语义。
    """

    def __init__(self, thread: threading.Thread | None) -> None:
        self._thread = thread
        self.stop_event = threading.Event()
        self.terminate_calls = 0

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.stop_event.set()


def make_session(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any = None,
    *,
    warmup_reply: tuple[str, str] = ("warmup", "fake engine"),
) -> WorkerSession:
    """构造注入 fake worker 的 WorkerSession。

    handler(task, cancel) -> WorkerOutcome：自定义单文件处理行为；
    warmup_reply：warmup 任务的应答（kind, note）。
    """
    session = WorkerSession()
    starts: list[AppConfig] = []

    def serve(
        task_queue: queue.Queue,
        outcome_queue: queue.Queue,
        events_queue: queue.Queue,
        cancel: threading.Event,
        ocr_cfg: OcrConfig,
        debug_cfg: DebugConfig,
    ) -> None:
        warmups = 0
        while True:
            try:
                task = task_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if task is None:
                return
            if task == _WARMUP_TASK:
                warmups += 1
                events_queue.put(("log", f"fake-warmup-{warmups}"))
                outcome_queue.put(warmup_reply)
                continue
            assert isinstance(task, WorkerTask)
            # 页事件先于结果返回（与真实 worker 的时序一致）。
            events_queue.put(("page", task.index, task.rel, 1, 1))
            if handler is not None:
                outcome_queue.put(handler(task, cancel))
            else:
                _write_fake_output(task)
                outcome_queue.put(_ok_outcome(task))

    def fake_start(cfg: AppConfig) -> None:
        _start_fake(session, cfg, handler=handler, warmup_reply=warmup_reply)
        starts.append(cfg)

    monkeypatch.setattr(session, "_start_worker", fake_start)
    session._test_starts = starts  # type: ignore[attr-defined]
    return session


def starts_of(session: WorkerSession) -> list[AppConfig]:
    return session._test_starts  # type: ignore[attr-defined]


def _write_fake_output(task: WorkerTask) -> None:
    Path(task.output).write_bytes(f"%PDF fake {task.rel}".encode("ascii"))


def _ok_outcome(
    task: WorkerTask,
    *,
    ocr_pages: int = 1,
    total_pages: int = 1,
    produced_output: bool = True,
) -> WorkerOutcome:
    return WorkerOutcome(
        index=task.index,
        rel=task.rel,
        kind="ok",
        total_pages=total_pages,
        ocr_pages=ocr_pages,
        all_existing_text=False,
        produced_output=produced_output,
        note="",
    )


def _outcome(task: WorkerTask, kind: str, *, note: str = "") -> WorkerOutcome:
    return WorkerOutcome(
        index=task.index,
        rel=task.rel,
        kind=kind,
        total_pages=0,
        ocr_pages=0,
        all_existing_text=False,
        produced_output=False,
        note=note,
    )


# ---------------------------------------------------------------------------
# 批次流程
# ---------------------------------------------------------------------------


def test_finalizes_outputs_in_scan_order(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.pdf").write_bytes(b"a")
    (tmp_path / "b.pdf").write_bytes(b"b")
    logs: list[str] = []
    progress: list[tuple[int, int, int, int, str]] = []
    file_results: list[FileResult] = []

    def handler(task: WorkerTask, cancel: threading.Event) -> WorkerOutcome:
        _write_fake_output(task)
        return _ok_outcome(task)

    session = make_session(monkeypatch, handler)

    summary = session.run_batch(
        [tmp_path],
        make_cfg(),
        None,
        progress_cb=lambda *args: progress.append(args),
        log_cb=logs.append,
        file_cb=file_results.append,
    )

    assert summary.cancelled is False
    assert [result.rel for result in summary.results] == ["a.pdf", "b.pdf"]
    assert [result.output.name for result in summary.results] == ["a_OCR.pdf", "b_OCR.pdf"]
    assert [result.rel for result in file_results] == ["a.pdf", "b.pdf"]
    assert {item[-1] for item in progress} == {"a.pdf", "b.pdf"}
    assert all(item[1] == 2 and item[2:4] == (1, 1) for item in progress)
    assert (tmp_path / "a_OCR.pdf").read_bytes() == b"%PDF fake a.pdf"
    assert any(line.startswith("OCR 完成: a.pdf") for line in logs)
    assert any(line.startswith("OCR 完成: b.pdf") for line in logs)
    # 一次只处理一个文件：结果顺序 = 扫描顺序。
    a_index = next(i for i, line in enumerate(logs) if line.startswith("OCR 完成: a.pdf"))
    b_index = next(i for i, line in enumerate(logs) if line.startswith("OCR 完成: b.pdf"))
    assert a_index < b_index
    assert list(tmp_path.rglob("~*.ocrtmp.pdf")) == []


def test_output_lands_next_to_source(tmp_path: Path, monkeypatch) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "picked.pdf").write_bytes(b"picked")
    (tmp_path / "outside.pdf").write_bytes(b"outside")
    session = make_session(monkeypatch)

    summary = session.run_batch([nested], make_cfg(), None)

    assert [result.rel for result in summary.results] == ["picked.pdf"]
    assert summary.results[0].status is FileStatus.SUCCESS_OCR
    assert summary.results[0].output == nested / "picked_OCR.pdf"
    assert summary.results[0].output is not None
    assert summary.results[0].output.read_bytes() == b"%PDF fake picked.pdf"
    assert not (tmp_path / "outside_OCR.pdf").exists()


def test_missing_explicit_file_fails_and_continues(tmp_path: Path, monkeypatch) -> None:
    present = tmp_path / "present.pdf"
    present.write_bytes(b"present")
    targets = [tmp_path / "missing.pdf", present]
    logs: list[str] = []

    def handler(task: WorkerTask, cancel: threading.Event) -> WorkerOutcome:
        if task.rel == "missing.pdf":
            # worker 对缺失文件显式检查后直接返回失败，不再触碰 OCR 引擎。
            return _outcome(task, "failed", note="文件不存在（可能已被移动或删除）")
        _write_fake_output(task)
        return _ok_outcome(task)

    session = make_session(monkeypatch, handler)

    summary = session.run_batch(targets, make_cfg(), None, log_cb=logs.append)

    assert [result.rel for result in summary.results] == ["missing.pdf", "present.pdf"]
    assert [result.status for result in summary.results] == [
        FileStatus.FAILED,
        FileStatus.SUCCESS_OCR,
    ]
    assert "文件不存在" in summary.results[0].note
    assert summary.results[1].output == tmp_path / "present_OCR.pdf"
    # 失败文件必须在运行日志中给出原因。
    assert any(line.startswith("失败: missing.pdf") for line in logs)


def test_cancel_stops_remaining_files(tmp_path: Path, monkeypatch) -> None:
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        (tmp_path / name).write_bytes(name.encode("ascii"))
    cancel_event = threading.Event()

    def handler(task: WorkerTask, cancel: threading.Event) -> WorkerOutcome:
        if task.rel == "c.pdf":
            # 页边界响应软取消：worker 收到 cancel 信号后返回 cancelled。
            while not cancel.is_set():
                time.sleep(0.005)
            return _outcome(task, "cancelled")
        _write_fake_output(task)
        result = _ok_outcome(task)
        if task.rel == "b.pdf":
            # b.pdf 处理完成后用户请求取消，c.pdf 尚未提交。
            cancel_event.set()
        return result

    session = make_session(monkeypatch, handler)

    summary = session.run_batch([tmp_path], make_cfg(), None, cancel_event=cancel_event)

    assert summary.cancelled is True
    assert [result.rel for result in summary.results] == ["a.pdf", "b.pdf"]
    assert all(result.output is not None for result in summary.results)
    assert not (tmp_path / "c_OCR.pdf").exists()
    assert list(tmp_path.rglob("~*.ocrtmp.pdf")) == []


def test_warmup_failure_marks_unskipped_files_failed(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.pdf").write_bytes(b"a")
    (tmp_path / "b.pdf").write_bytes(b"b")
    file_results: list[FileResult] = []
    logs: list[str] = []

    session = make_session(
        monkeypatch,
        warmup_reply=("warmup_error", "RuntimeError: model missing"),
    )

    summary = session.run_batch(
        [tmp_path],
        make_cfg(),
        None,
        log_cb=logs.append,
        file_cb=file_results.append,
    )

    assert summary.cancelled is False
    assert [result.status for result in summary.results] == [FileStatus.FAILED, FileStatus.FAILED]
    assert [result.rel for result in file_results] == ["a.pdf", "b.pdf"]
    assert all("RuntimeError: model missing" in result.note for result in summary.results)
    # 模型加载失败也要在运行日志中给出每个文件的原因。
    assert logs == [
        "失败: a.pdf（RuntimeError: model missing）",
        "失败: b.pdf（RuntimeError: model missing）",
    ]


def test_warmup_worker_death_fails_batch(tmp_path: Path, monkeypatch) -> None:
    """warmup 阶段 worker 进程死亡：批次立即失败，不挂死在「加载中」。

    Windows 上父端 queue.get 在对端死亡后永远 Empty（不会 EOFError），
    死亡检测必须依赖进程存活检查——本用例守护这条路径。
    """
    (tmp_path / "a.pdf").write_bytes(b"a")
    logs: list[str] = []

    def dying_during_warmup(
        task_queue: queue.Queue,
        outcome_queue: queue.Queue,
        events_queue: queue.Queue,
        cancel: threading.Event,
        ocr_cfg: OcrConfig,
        debug_cfg: DebugConfig,
    ) -> None:
        assert task_queue.get(timeout=2) == _WARMUP_TASK
        # 不回任何结果直接退出：模拟 spawn 失败 / segfault。

    session = WorkerSession()

    def fake_start(cfg: AppConfig) -> None:
        task_q: queue.Queue = queue.Queue()
        out_q: queue.Queue = queue.Queue()
        ev_q: queue.Queue = queue.Queue()
        cancel = threading.Event()
        thread = threading.Thread(
            target=dying_during_warmup,
            args=(task_q, out_q, ev_q, cancel, cfg.ocr, cfg.debug),
            daemon=True,
        )
        thread.start()
        session._task_queue = task_q
        session._outcome_queue = out_q
        session._events_queue = ev_q
        session._worker_cancel = cancel
        session._process = _FakeProcess(thread)
        session._worker_ocr_cfg = cfg.ocr
        session._worker_debug_cfg = cfg.debug
        session._engine_ready = False

    monkeypatch.setattr(session, "_start_worker", fake_start)

    summary = session.run_batch([tmp_path], make_cfg(), None, log_cb=logs.append)

    assert summary.cancelled is False
    assert [result.status for result in summary.results] == [FileStatus.FAILED]
    assert "工作进程意外退出" in summary.results[0].note
    assert session._engine_ready is False


def test_pending_cb_lists_every_file_before_processing(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.pdf").write_bytes(b"a")
    (tmp_path / "b.pdf").write_bytes(b"b")
    pending: list[list[str]] = []
    session = make_session(monkeypatch)

    summary = session.run_batch(
        [tmp_path],
        make_cfg(),
        None,
        pending_cb=pending.append,
    )

    assert pending == [["a.pdf", "b.pdf"]]
    assert [result.rel for result in summary.results] == ["a.pdf", "b.pdf"]


def test_all_files_skipped_does_not_start_worker(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "done.pdf"
    source.write_bytes(b"done")
    output = tmp_path / "done_OCR.pdf"
    output.write_bytes(b"previous")

    session = make_session(monkeypatch)
    monkeypatch.setattr(
        session,
        "_ensure_worker",
        lambda cfg: (_ for _ in ()).throw(AssertionError("no worker should start")),
    )

    summary = session.run_batch([tmp_path], make_cfg(), None)

    assert [result.status for result in summary.results] == [FileStatus.SKIPPED_DONE]
    assert summary.results[0].output == output
    assert starts_of(session) == []


def test_overwrite_mode_skips_existing_image_output(tmp_path: Path, monkeypatch) -> None:
    # 图片不能覆盖原文件：_OCR 已存在时不启动 worker、不重复 OCR。
    source = tmp_path / "done.png"
    source.write_bytes(b"image bytes")
    output = tmp_path / "done_OCR.pdf"
    output.write_bytes(b"previous")

    session = make_session(monkeypatch)
    monkeypatch.setattr(
        session,
        "_ensure_worker",
        lambda cfg: (_ for _ in ()).throw(AssertionError("no worker should start")),
    )

    cfg = make_cfg(output=OutputConfig(overwrite_original=True))
    summary = session.run_batch([tmp_path], cfg, None)

    assert [result.status for result in summary.results] == [FileStatus.SKIPPED_DONE]
    assert summary.results[0].output == output


def test_warns_when_medium_model_selected(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.pdf").write_bytes(b"a")
    logs: list[str] = []
    slow_cfg = make_cfg(ocr=OcrConfig(ocr_version="PP-OCRv6", model_type="medium"))
    session = make_session(monkeypatch)

    session.run_batch([tmp_path], slow_cfg, None, log_cb=logs.append)

    warnings = [message for message in logs if message.startswith("提示：")]
    assert len(warnings) == 1
    assert "PP-OCRv6 medium" in warnings[0]


def test_empty_folder_reports_no_files(tmp_path: Path, monkeypatch) -> None:
    logs: list[str] = []
    session = make_session(monkeypatch)
    monkeypatch.setattr(
        session,
        "_ensure_worker",
        lambda cfg: (_ for _ in ()).throw(AssertionError("empty input must not start worker")),
    )

    summary = session.run_batch([tmp_path], make_cfg(), None, log_cb=logs.append)

    assert summary.results == []
    assert logs == ["找不到可处理的 PDF/图片文件。"]


def test_skip_logs_come_after_warmup_done(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "done.pdf").write_bytes(b"d")
    (tmp_path / "done_OCR.pdf").write_bytes(b"out")
    (tmp_path / "new.pdf").write_bytes(b"n")
    logs: list[str] = []
    session = make_session(monkeypatch)

    summary = session.run_batch([tmp_path], make_cfg(), None, log_cb=logs.append)

    assert summary.cancelled is False
    assert Counter(result.status for result in summary.results) == Counter(
        {FileStatus.SUCCESS_OCR: 1, FileStatus.SKIPPED_DONE: 1}
    )
    # 跳过检查虽然在模型确认前执行，日志必须延后到"加载完成"之后输出，
    # 保证运行日志按用户预期的流程排序（先加载，后处理结果）。
    done_index = next(i for i, line in enumerate(logs) if "加载完成" in line)
    skip_index = next(i for i, line in enumerate(logs) if "跳过（已输出）" in line)
    assert done_index < skip_index


def test_encrypted_subfolder_reports_without_output(tmp_path: Path, monkeypatch) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "secret.pdf").write_bytes(b"secret")

    def handler(task: WorkerTask, cancel: threading.Event) -> WorkerOutcome:
        return _outcome(task, "encrypted", note="encrypted")

    session = make_session(monkeypatch, handler)

    summary = session.run_batch(
        [tmp_path],
        make_cfg(input=InputConfig(recursive=True)),
        None,
    )

    assert [result.rel for result in summary.results] == ["nested/secret.pdf"]
    assert summary.results[0].status is FileStatus.SKIPPED_ENCRYPTED
    assert summary.results[0].output is None
    assert not (nested / "secret_OCR.pdf").exists()


# ---------------------------------------------------------------------------
# 引擎常驻：模型只加载一次；配置变化才重启
# ---------------------------------------------------------------------------


def test_engine_warms_up_once_across_batches(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.pdf").write_bytes(b"a")
    (tmp_path / "b.pdf").write_bytes(b"b")
    logs: list[str] = []
    session = make_session(monkeypatch)

    session.run_batch([tmp_path / "a.pdf"], make_cfg(), None, log_cb=logs.append)
    session.run_batch([tmp_path / "b.pdf"], make_cfg(), None, log_cb=logs.append)

    # 只启动一次 worker 进程。
    assert len(starts_of(session)) == 1
    # 第一批走加载完成文案，第二批模型已就绪、不再等待加载。
    assert len([line for line in logs if "加载完成" in line]) == 1
    assert any("已就绪" in line for line in logs)


def test_config_change_restarts_worker_and_reloads(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.pdf").write_bytes(b"a")
    logs: list[str] = []
    session = make_session(monkeypatch)

    session.run_batch([tmp_path], make_cfg(), None, log_cb=logs.append)
    changed = make_cfg(ocr=OcrConfig(dpi=300))
    session.run_batch([tmp_path], changed, None, log_cb=logs.append, force=True)

    # 配置变化：重启进程（模型必须重新加载），共两次启动、两次加载。
    assert len(starts_of(session)) == 2
    assert len([line for line in logs if "加载完成" in line]) == 2


def test_stale_temp_files_are_cleaned_up(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.pdf").write_bytes(b"a")
    stale = tmp_path / "~deadbeef.ocrtmp.pdf"
    stale.write_bytes(b"stale from v0.8 crash")
    session = make_session(monkeypatch)

    summary = session.run_batch([tmp_path], make_cfg(), None)

    assert summary.cancelled is False
    # 旧版本遗留的临时文件批次开始时被清理，且新实现不再产生临时文件。
    assert not stale.exists()
    assert list(tmp_path.rglob("~*.ocrtmp.pdf")) == []


def test_worker_death_marks_remaining_files_failed(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.pdf").write_bytes(b"a")
    (tmp_path / "b.pdf").write_bytes(b"b")
    (tmp_path / "c.pdf").write_bytes(b"c")
    logs: list[str] = []
    file_results: list[FileResult] = []

    def dying_serve(
        task_queue: queue.Queue,
        outcome_queue: queue.Queue,
        events_queue: queue.Queue,
        cancel: threading.Event,
        ocr_cfg: OcrConfig,
        debug_cfg: DebugConfig,
    ) -> None:
        # 预热后处理第一个文件，然后线程直接结束，模拟子进程崩溃。
        warmup_task = task_queue.get(timeout=2)
        assert warmup_task == _WARMUP_TASK
        outcome_queue.put(("warmup", "fake engine"))
        task = task_queue.get(timeout=2)
        _write_fake_output(task)
        outcome_queue.put(_ok_outcome(task))

    session = WorkerSession()

    def fake_start(cfg: AppConfig) -> None:
        task_q: queue.Queue = queue.Queue()
        out_q: queue.Queue = queue.Queue()
        ev_q: queue.Queue = queue.Queue()
        cancel = threading.Event()
        thread = threading.Thread(
            target=dying_serve,
            args=(task_q, out_q, ev_q, cancel, cfg.ocr, cfg.debug),
            daemon=True,
        )
        thread.start()
        session._task_queue = task_q
        session._outcome_queue = out_q
        session._events_queue = ev_q
        session._worker_cancel = cancel
        session._process = _FakeProcess(thread)
        session._worker_ocr_cfg = cfg.ocr
        session._worker_debug_cfg = cfg.debug
        session._engine_ready = False

    monkeypatch.setattr(session, "_start_worker", fake_start)

    summary = session.run_batch(
        [tmp_path],
        make_cfg(),
        None,
        log_cb=logs.append,
        file_cb=file_results.append,
    )

    assert summary.cancelled is False
    assert [result.rel for result in summary.results] == ["a.pdf", "b.pdf", "c.pdf"]
    assert summary.results[0].status is FileStatus.SUCCESS_OCR
    assert [result.status for result in summary.results[1:]] == [
        FileStatus.FAILED,
        FileStatus.FAILED,
    ]
    assert all("工作进程意外退出" in result.note for result in summary.results[1:])
    assert [result.rel for result in file_results] == ["a.pdf", "b.pdf", "c.pdf"]


# ---------------------------------------------------------------------------
# 取消与关闭
# ---------------------------------------------------------------------------


def test_cancel_during_warmup_stops_batch_and_shuts_down_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "a.pdf").write_bytes(b"a")
    cancel_event = threading.Event()

    def blocking_warmup_serve(
        task_queue: queue.Queue,
        outcome_queue: queue.Queue,
        events_queue: queue.Queue,
        cancel: threading.Event,
        ocr_cfg: OcrConfig,
        debug_cfg: DebugConfig,
    ) -> None:
        while True:
            try:
                task = task_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if task is None:
                return
            if task == _WARMUP_TASK:
                # 模拟模型加载：C++ 调用无法中断，只能等协调器终止进程。
                deadline = time.monotonic() + 10
                while not cancel.is_set() and time.monotonic() < deadline:
                    time.sleep(0.01)
                return
            _write_fake_output(task)
            outcome_queue.put(_ok_outcome(task))

    session = WorkerSession()

    def fake_start(cfg: AppConfig) -> None:
        task_q: queue.Queue = queue.Queue()
        out_q: queue.Queue = queue.Queue()
        ev_q: queue.Queue = queue.Queue()
        cancel = threading.Event()
        thread = threading.Thread(
            target=blocking_warmup_serve,
            args=(task_q, out_q, ev_q, cancel, cfg.ocr, cfg.debug),
            daemon=True,
        )
        thread.start()
        session._task_queue = task_q
        session._outcome_queue = out_q
        session._events_queue = ev_q
        session._worker_cancel = cancel
        session._process = _FakeProcess(thread)
        session._worker_ocr_cfg = cfg.ocr
        session._worker_debug_cfg = cfg.debug
        session._engine_ready = False

    monkeypatch.setattr(session, "_start_worker", fake_start)

    def cancel_soon() -> None:
        time.sleep(0.15)
        cancel_event.set()

    watchdog = threading.Thread(target=cancel_soon)
    watchdog.start()
    summary = session.run_batch([tmp_path], make_cfg(), None, cancel_event=cancel_event)
    watchdog.join()

    assert summary.cancelled is True
    assert summary.results == []
    # 加载中被取消：worker 进程被终止，引擎状态复位。
    assert session._process is None
    assert not (tmp_path / "a_OCR.pdf").exists()


def test_page_events_arrive_before_file_done(tmp_path: Path, monkeypatch) -> None:
    """页事件必须先于该文件的 file_done 转发，且之后不再有该文件的页事件。"""
    (tmp_path / "a.pdf").write_bytes(b"a")
    (tmp_path / "b.pdf").write_bytes(b"b")
    order: list[str] = []

    def progress_cb(file_i, file_n, page_i, page_n, name):
        order.append(f"page:{name}")

    def file_cb(result: FileResult) -> None:
        order.append(f"file:{result.rel}")

    session = make_session(monkeypatch)
    session.run_batch([tmp_path], make_cfg(), None, progress_cb=progress_cb, file_cb=file_cb)

    assert order == ["page:a.pdf", "file:a.pdf", "page:b.pdf", "file:b.pdf"]


def test_session_shutdown_stops_worker(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.pdf").write_bytes(b"a")
    session = make_session(monkeypatch)

    session.run_batch([tmp_path], make_cfg(), None)
    process = session._process
    assert process is not None
    thread = process._thread  # type: ignore[attr-defined]

    session.shutdown()

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert session._process is None
    # 待命进程走优雅退出（退出哨兵生效，join 内自然结束），没有强杀。
    assert process.terminate_calls == 0
    assert session._in_flight_temps == {}


def test_shutdown_is_idempotent(monkeypatch) -> None:
    session = make_session(monkeypatch)
    session.shutdown()
    session.shutdown()
    assert session._process is None


def test_cancel_then_immediate_restart_reuses_engine(tmp_path: Path, monkeypatch) -> None:
    """取消后立即重新开始：取消信号必须复位、引擎复用不重载、文件正常处理。"""
    for name in ("a.pdf", "b.pdf", "c.pdf", "d.pdf"):
        (tmp_path / name).write_bytes(name.encode("ascii"))
    logs: list[str] = []
    first_cancel = threading.Event()

    def handler(task: WorkerTask, cancel: threading.Event) -> WorkerOutcome:
        if task.rel == "c.pdf":
            while not cancel.is_set():
                time.sleep(0.005)
            return _outcome(task, "cancelled")
        _write_fake_output(task)
        result = _ok_outcome(task)
        if task.rel == "b.pdf":
            # b 处理完即请求取消，c 在页边界响应软取消。
            first_cancel.set()
        return result

    session = make_session(monkeypatch, handler)
    summary = session.run_batch(
        [tmp_path],
        make_cfg(),
        None,
        cancel_event=first_cancel,
        log_cb=logs.append,
    )
    assert summary.cancelled is True

    # 立即重新开始（GUI 每次开始都新建取消事件）。
    second_cancel = threading.Event()
    summary2 = session.run_batch(
        [tmp_path / "d.pdf"],
        make_cfg(),
        None,
        cancel_event=second_cancel,
        log_cb=logs.append,
    )

    # 没有重启进程：引擎复用，不重新加载模型。
    assert len(starts_of(session)) == 1
    assert len([line for line in logs if "加载完成" in line]) == 1
    assert any("已就绪" in line for line in logs)
    # 上一批残留的取消信号已复位：新批次的文件正常完成，而不是秒取消。
    assert summary2.cancelled is False
    assert [result.status for result in summary2.results] == [FileStatus.SUCCESS_OCR]


def test_stale_outcomes_are_drained_before_new_batch(tmp_path: Path, monkeypatch) -> None:
    """上一批残留的结果消息不得被误配到新批次任务。"""
    (tmp_path / "a.pdf").write_bytes(b"a")
    session = make_session(monkeypatch)
    # 先建好队列（等价于上一批已启动 worker），再塞入上一批的旧结果。
    session._start_worker(make_cfg())
    stale = WorkerOutcome(
        index=0,
        rel="a.pdf",
        kind="failed",
        total_pages=0,
        ocr_pages=0,
        all_existing_text=False,
        produced_output=False,
        note="stale from previous batch",
    )
    session._outcome_queue.put(stale)

    summary = session.run_batch([tmp_path], make_cfg(), None)

    assert [result.status for result in summary.results] == [FileStatus.SUCCESS_OCR]


def test_in_flight_temp_removed_when_worker_dies(tmp_path: Path, monkeypatch) -> None:
    """worker 写到一半崩溃：协调器用任务自带的临时路径精确清理残留。"""
    (tmp_path / "a.pdf").write_bytes(b"a")
    logs: list[str] = []

    def crashing_serve(
        task_queue: queue.Queue,
        outcome_queue: queue.Queue,
        events_queue: queue.Queue,
        cancel: threading.Event,
        ocr_cfg: OcrConfig,
        debug_cfg: DebugConfig,
    ) -> None:
        assert task_queue.get(timeout=2) == _WARMUP_TASK
        outcome_queue.put(("warmup", "fake engine"))
        task = task_queue.get(timeout=2)
        assert isinstance(task, WorkerTask)
        # 模拟 save 中途崩溃：临时文件遗留磁盘、outcome 永远不会送达。
        Path(task.temp_output).write_bytes(b"%PDF half-written")

    session = WorkerSession()

    def fake_start(cfg: AppConfig) -> None:
        # 自定义整循环 serve：直接以线程运行（其自身负责退出时机）。
        task_q: queue.Queue = queue.Queue()
        out_q: queue.Queue = queue.Queue()
        ev_q: queue.Queue = queue.Queue()
        cancel = threading.Event()
        thread = threading.Thread(
            target=crashing_serve,
            args=(task_q, out_q, ev_q, cancel, cfg.ocr, cfg.debug),
            daemon=True,
        )
        thread.start()
        session._task_queue = task_q
        session._outcome_queue = out_q
        session._events_queue = ev_q
        session._worker_cancel = cancel
        session._process = _FakeProcess(thread)
        session._worker_ocr_cfg = cfg.ocr
        session._worker_debug_cfg = cfg.debug
        session._engine_ready = False

    monkeypatch.setattr(session, "_start_worker", fake_start)

    summary = session.run_batch([tmp_path], make_cfg(), None, log_cb=logs.append)

    assert [result.status for result in summary.results] == [FileStatus.FAILED]
    assert "工作进程意外退出" in summary.results[0].note
    # 协调器按任务自带的临时路径清理，目录里不留 ~*.ocrtmp.pdf。
    assert list(tmp_path.glob("~*.ocrtmp.pdf")) == []


def test_shutdown_cleans_in_flight_temp(tmp_path: Path, monkeypatch) -> None:
    """批次进行中关闭程序：正在写的临时文件被 shutdown 精确清理。"""
    (tmp_path / "a.pdf").write_bytes(b"a")
    temp_seen = threading.Event()

    def slow_serve(
        task_queue: queue.Queue,
        outcome_queue: queue.Queue,
        events_queue: queue.Queue,
        cancel: threading.Event,
        ocr_cfg: OcrConfig,
        debug_cfg: DebugConfig,
    ) -> None:
        if task_queue.get(timeout=2) == _WARMUP_TASK:
            outcome_queue.put(("warmup", "fake engine"))
        task = task_queue.get(timeout=2)
        assert isinstance(task, WorkerTask)
        Path(task.temp_output).write_bytes(b"%PDF half-written")
        temp_seen.set()
        while not cancel.is_set():
            time.sleep(0.01)

    session = WorkerSession()

    def fake_start(cfg: AppConfig) -> None:
        # 自定义整循环 serve：直接以线程运行（其自身负责退出时机）。
        task_q: queue.Queue = queue.Queue()
        out_q: queue.Queue = queue.Queue()
        ev_q: queue.Queue = queue.Queue()
        cancel = threading.Event()
        thread = threading.Thread(
            target=slow_serve,
            args=(task_q, out_q, ev_q, cancel, cfg.ocr, cfg.debug),
            daemon=True,
        )
        thread.start()
        session._task_queue = task_q
        session._outcome_queue = out_q
        session._events_queue = ev_q
        session._worker_cancel = cancel
        session._process = _FakeProcess(thread)
        session._worker_ocr_cfg = cfg.ocr
        session._worker_debug_cfg = cfg.debug
        session._engine_ready = False

    monkeypatch.setattr(session, "_start_worker", fake_start)

    batch = threading.Thread(
        target=session.run_batch,
        args=([tmp_path], make_cfg(), None),
        daemon=True,
    )
    batch.start()
    assert temp_seen.wait(timeout=5)
    leftover = list(tmp_path.glob("~*.ocrtmp.pdf"))
    assert len(leftover) == 1  # worker 留下的半截临时文件在磁盘上

    session.shutdown()
    batch.join(timeout=10)

    assert not leftover[0].exists()
    assert session._process is None


def _start_fake(
    session: WorkerSession,
    cfg: AppConfig,
    *,
    handler: Any = None,
    warmup_reply: tuple[str, str] = ("warmup", "fake engine"),
) -> None:
    """把默认 fake worker（线程版）挂到 session 上（替代 spawn 子进程）。

    线程循环以 _FakeProcess.stop_event 为存活条件：terminate 后退出，
    模拟真进程被终止的语义；软取消（worker_cancel）不会停掉循环。
    """
    task_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    ev_q: queue.Queue = queue.Queue()
    cancel = threading.Event()
    process = _FakeProcess(None)

    def target() -> None:
        warmups = 0
        while not process.stop_event.is_set():
            try:
                task = task_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if task is None:
                return
            if task == _WARMUP_TASK:
                warmups += 1
                ev_q.put(("log", f"fake-warmup-{warmups}"))
                out_q.put(warmup_reply)
                continue
            assert isinstance(task, WorkerTask)
            # 页事件先于结果返回（与真实 worker 的时序一致）。
            ev_q.put(("page", task.index, task.rel, 1, 1))
            if handler is not None:
                out_q.put(handler(task, cancel))
            else:
                _write_fake_output(task)
                out_q.put(_ok_outcome(task))

    thread = threading.Thread(target=target, daemon=True)
    process._thread = thread
    thread.start()
    session._task_queue = task_q
    session._outcome_queue = out_q
    session._events_queue = ev_q
    session._worker_cancel = cancel
    session._process = process
    session._worker_ocr_cfg = cfg.ocr
    session._worker_debug_cfg = cfg.debug
    session._engine_ready = False


def test_no_text_outcome_produces_no_output_and_never_overwrites(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """OCR 未识别出任何文字：不产出输出文件，覆盖模式也绝不触发。"""
    (tmp_path / "scan.pdf").write_bytes(b"scan")
    overwrite_calls: list[tuple[Path, Path]] = []

    def handler(task: WorkerTask, cancel: threading.Event) -> WorkerOutcome:
        return _ok_outcome(task, ocr_pages=0, produced_output=False)

    session = make_session(monkeypatch, handler)
    monkeypatch.setattr(
        "pdf_ocrer.runner._overwrite_after_finalize",
        lambda src, output: overwrite_calls.append((src, output)) or None,
    )
    cfg = make_cfg(output=OutputConfig(overwrite_original=True))

    summary = session.run_batch([tmp_path], cfg, None)

    result = summary.results[0]
    assert result.status is FileStatus.NO_TEXT_FOUND
    assert result.output is None
    assert not (tmp_path / "scan_OCR.pdf").exists()
    assert overwrite_calls == []
