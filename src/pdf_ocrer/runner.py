"""Single long-lived worker process for sequential OCR batches.

Replaces the former multi-process parallel coordinator. Key properties:

- **Process isolation**: ONNX inference does not release the GIL, so running
  it inside the Tk process freezes the UI. OCR therefore runs in a separate
  ``spawn`` process while the GUI stays responsive.
- **Engine residency**: the worker process is started lazily on the first
  batch and stays alive until the app closes, so the model is loaded once per
  run instead of once per batch. Changing the OCR config restarts the process
  (the model must be reloaded anyway).
- **One file at a time**: tasks are submitted sequentially, one per wait.
- **Crash-safe output**: the worker saves to a same-folder ``~*.ocrtmp.pdf``
  temp file and atomically renames it into the final ``原名_OCR.pdf``, so the
  output is either absent or complete — a mid-write failure or a killed
  process never leaves a half-written file that the "output already exists"
  check would skip forever. Crash leftovers are cleaned up when the next
  batch starts.
"""

from __future__ import annotations

import logging
import multiprocessing
import queue as queue_mod
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from pdf_ocrer.config import AppConfig, ConfigError
from pdf_ocrer.deps import ensure_engine_deps
from pdf_ocrer.ocr_engine import OcrEngineProtocol
from pdf_ocrer.pipeline import (
    BatchSummary,
    FileResult,
    FileStatus,
    ProgressCb,
    _file_log_message,
    _log_batch_end,
    _log_file_result,
    _overwrite_after_finalize,
    _should_skip_existing_output,
    emit_slow_model_warning,
    expand_targets,
)
from pdf_ocrer.scanning import OUTPUT_SUFFIX
from pdf_ocrer.worker import WARMUP_TASK, WorkerOutcome, WorkerTask, worker_serve

_logger = logging.getLogger(__name__)

# 软取消宽限：worker 在页边界响应取消信号的最后期限。取消时往往正在
# 推理一页（CPU 上单页可达 10-20 秒），宽限必须覆盖整页推理时间，
# 否则就会硬杀常驻进程、丢失已加载的引擎（下批次被迫重新加载模型）。
_CANCEL_SOFT_GRACE_SECONDS = 15.0

_WORKER_DEAD_NOTE = "工作进程意外退出，文件未处理"


@dataclass(frozen=True)
class _PendingFile:
    index: int
    source: Path
    rel: str
    output: Path
    task: WorkerTask


class WorkerSession:
    """常驻 OCR 工作子进程的协调器；生命周期与 GUI 一致。

    进程懒启动：第一次批次时 spawn，之后批次复用（模型已加载，免等待），
    程序关闭时 :meth:`shutdown` 释放。OCR 配置变化时重启进程。
    """

    def __init__(self) -> None:
        self._process: Any | None = None
        self._task_queue: Any | None = None
        self._outcome_queue: Any | None = None
        self._events_queue: Any | None = None
        self._worker_cancel: Any | None = None
        self._worker_ocr_cfg: Any | None = None
        self._worker_debug_cfg: Any | None = None
        self._engine_ready = False
        self._warmup_elapsed = 0.0
        # shutdown 可能来自关闭窗口的后台线程，也可能来自批次线程的
        # 取消超时路径：加锁防止两个线程并发终止同一进程。
        self._shutdown_lock = threading.Lock()
        # 进行中任务的临时文件（index → 路径）：由协调器生成并持有，
        # worker 死亡/终止/关闭程序时精确清理，不依赖"下次批次"兜底。
        self._in_flight_temps: dict[int, Path] = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def _ensure_worker(self, cfg: AppConfig) -> bool:
        """保证工作子进程存活且配置一致。

        返回 True 表示（重）启动了新进程——引擎必然未加载。
        """
        alive = self._process is not None and self._process.is_alive()
        if (
            alive
            and self._worker_ocr_cfg == cfg.ocr
            and self._worker_debug_cfg == cfg.debug
        ):
            return False

        if alive:
            # 配置变化：模型必须重新加载，重启进程而不是让 worker 热替换。
            _logger.info("ocr config changed, restarting worker process")
            self.shutdown()
        self._start_worker(cfg)
        return True

    def _start_worker(self, cfg: AppConfig) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._task_queue = ctx.Queue()
        self._outcome_queue = ctx.Queue()
        self._events_queue = ctx.Queue()
        self._worker_cancel = ctx.Event()
        self._process = ctx.Process(
            target=worker_serve,
            args=(
                self._task_queue,
                self._outcome_queue,
                self._events_queue,
                self._worker_cancel,
                cfg.ocr,
                cfg.debug,
            ),
            daemon=True,
        )
        self._process.start()
        self._worker_ocr_cfg = cfg.ocr
        self._worker_debug_cfg = cfg.debug
        self._engine_ready = False

    def shutdown(self) -> None:
        """停止工作子进程：退出哨兵 → 短暂等待 → terminate → kill。

        进程结束后精确清理未完成任务的临时文件，不留残留。
        加锁串行化：关闭窗口的后台线程与批次线程的取消超时路径可能同时
        进入，避免并发 join/terminate 同一进程。
        """
        with self._shutdown_lock:
            process = self._process
            self._process = None
            self._engine_ready = False
            if process is None:
                self._discard_pending_temps()
                return

            try:
                if process.is_alive():
                    # 优雅退出：先投退出哨兵（worker 主循环收到 None 即结束），
                    # 避免每次关闭都等到 join 超时后强杀（窗口"慢一拍"的原因）。
                    if self._task_queue is not None:
                        try:
                            self._task_queue.put_nowait(None)
                        except (OSError, ValueError, queue_mod.Full):
                            pass
                    if self._worker_cancel is not None:
                        self._worker_cancel.set()
                    # 短暂等待收尾即可：处理中的取消等待已由 _process_one 的
                    # 软取消宽限负责；模型加载中的取消反正无法中断，不必长等。
                    process.join(2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(1.0)
                if process.is_alive():
                    process.kill()
                    # Windows 的 kill 是异步的：等句柄释放，避免随后的
                    # 临时文件清理因句柄未关而 PermissionError。
                    process.join(1.0)
            except (OSError, ValueError):
                _logger.warning("worker shutdown failed", exc_info=True)
            finally:
                for q in (self._task_queue, self._outcome_queue, self._events_queue):
                    if q is None:
                        continue
                    close = getattr(q, "close", None)
                    if close is None:
                        continue
                    try:
                        close()
                    except (OSError, ValueError):
                        pass
            # 进程已终止：精确清理来不及收尾的临时文件。
            self._discard_pending_temps()
            self._task_queue = None
            self._outcome_queue = None
            self._events_queue = None
            self._worker_cancel = None
            self._worker_ocr_cfg = None
            self._worker_debug_cfg = None

    # ------------------------------------------------------------------
    # 批次流程
    # ------------------------------------------------------------------

    def run_batch(
        self,
        targets: list[Path],
        cfg: AppConfig,
        engine: OcrEngineProtocol | None,
        progress_cb: ProgressCb | None = None,
        log_cb: Callable[[str], None] | None = None,
        cancel_event: Any | None = None,
        file_cb: Callable[[FileResult], None] | None = None,
        pending_cb: Callable[[list[str]], None] | None = None,
        page_start_cb: ProgressCb | None = None,
        force: bool = False,
    ) -> BatchSummary:
        _ = engine  # 签名兼容：OCR 引擎由常驻子进程持有，协调器不加载。

        items = expand_targets(targets, cfg.input)
        total_files = len(items)
        _logger.info(
            "batch start targets=%s files=%d",
            [str(path) for path in targets],
            len(items),
        )
        # 先把全部待处理文件告诉界面，让表格在开始处理时列出完整清单。
        if pending_cb is not None:
            pending_cb([item.rel for item in items])

        if not items:
            if log_cb is not None:
                log_cb("找不到可处理的 PDF/图片文件。")
            summary = BatchSummary(results=[], cancelled=False)
            _log_batch_end(summary)
            return summary

        # 慢模型提前说清楚，避免用户把整页推理时间误认为"卡在加载模型"。
        emit_slow_model_warning(cfg, log_cb)

        # 协调器侧依赖检查：保证依赖就绪（需要时安装一次），
        # 避免 worker 进程互相竞争去 import 引擎。
        try:
            ensure_engine_deps("rapidocr", log_cb)
        except ConfigError as exc:
            if log_cb is not None:
                log_cb(str(exc))
            summary = BatchSummary(results=[], cancelled=False)
            _log_batch_end(summary)
            return summary

        result_by_index: dict[int, FileResult] = {}
        pending_by_index: dict[int, _PendingFile] = {}
        cancelled = False

        # 处理中途崩溃残留的临时文件（~*.ocrtmp.pdf）：批次开始前顺手清理，
        # 避免残留堆积；正式输出经原子改名落盘，不受清理影响。
        _cleanup_stale_temps({item.src.parent for item in items})

        def emit_result(index: int, result: FileResult) -> None:
            result_by_index[index] = result
            _log_file_result(result)
            message = _file_log_message(result)
            if message is not None and log_cb is not None:
                log_cb(message)
            if file_cb is not None:
                file_cb(result)

        # 跳过检查在模型确认前执行，但日志延后到模型就绪后再输出，
        # 保证运行日志顺序与用户预期的流程一致（先加载，后处理结果）。
        skipped_logs: list[str] = []

        def _flush_skipped_logs() -> None:
            if log_cb is not None:
                for message in skipped_logs:
                    log_cb(message)
            skipped_logs.clear()

        for index, item in enumerate(items):
            if _event_is_set(cancel_event):
                cancelled = True
                break

            output = item.src.parent / f"{item.src.stem}{OUTPUT_SUFFIX}.pdf"
            if _should_skip_existing_output(item.src, output, cfg, force):
                result = FileResult(
                    source=item.src,
                    output=output,
                    status=FileStatus.SKIPPED_DONE,
                    total_pages=0,
                    ocr_pages=0,
                    note="",
                    rel=item.rel,
                )
                result_by_index[index] = result
                _log_file_result(result)
                skipped_logs.append(f"跳过（已输出）: {item.rel}")
                _logger.info("跳过（已输出）: %s", item.rel)
                if file_cb is not None:
                    file_cb(result)
                continue

            temp_output = item.src.parent / f"~{uuid4().hex}.ocrtmp.pdf"
            pending_by_index[index] = _PendingFile(
                index=index,
                source=item.src,
                rel=item.rel,
                output=output,
                task=WorkerTask(
                    index=index,
                    source=str(item.src),
                    rel=item.rel,
                    output=str(output),
                    temp_output=str(temp_output),
                ),
            )

        if cancelled or not pending_by_index:
            _flush_skipped_logs()
            summary = BatchSummary(
                results=_ordered_results(result_by_index),
                cancelled=cancelled,
            )
            _log_batch_end(summary)
            return summary

        if self._ensure_worker(cfg) or not self._engine_ready:
            # 每批次开始必须复位取消信号：上一批的软取消会把 worker_cancel
            # 置位，不清除的话 worker 一进处理就自我取消，批次"无法启动"。
            if self._worker_cancel is not None:
                self._worker_cancel.clear()
            self._drain_outcomes()
            warmup_status, warmup_note = self._warmup(
                cancel_event, progress_cb, page_start_cb, log_cb, total_files
            )
            if warmup_status == "cancelled":
                cancelled = True
                _flush_skipped_logs()
                summary = BatchSummary(
                    results=_ordered_results(result_by_index),
                    cancelled=True,
                )
                _log_batch_end(summary)
                return summary
            if warmup_status == "failed":
                _logger.error("OCR warmup failed: %s", warmup_note)
                _flush_skipped_logs()
                for index in sorted(pending_by_index):
                    emit_result(index, _failed_result(pending_by_index[index], warmup_note))
                summary = BatchSummary(
                    results=_ordered_results(result_by_index),
                    cancelled=False,
                )
                _log_batch_end(summary)
                return summary
            _log_message(
                f"OCR 模型加载完成（用时 {self._warmup_elapsed:.1f} 秒），开始识别…",
                log_cb,
            )
            self._engine_ready = True
        else:
            # 复用常驻进程：同样要复位取消信号并排空上一批的残留结果，
            # 否则上一批的取消信号会让本批所有文件立即自我取消。
            if self._worker_cancel is not None:
                self._worker_cancel.clear()
            self._drain_outcomes()
            _log_message("OCR 模型已就绪，开始处理…", log_cb)

        _flush_skipped_logs()
        self._drain_events(progress_cb, page_start_cb, log_cb, total_files, timeout=0.0)

        if _event_is_set(cancel_event):
            cancelled = True
            summary = BatchSummary(
                results=_ordered_results(result_by_index),
                cancelled=True,
            )
            _log_batch_end(summary)
            return summary

        ordered_indices = sorted(pending_by_index)
        for position, index in enumerate(ordered_indices):
            pending = pending_by_index[index]
            outcome = self._process_one(
                pending.task,
                cancel_event,
                progress_cb,
                page_start_cb,
                log_cb,
                total_files,
            )
            if outcome is None:
                if _event_is_set(cancel_event):
                    cancelled = True
                else:
                    # 工作进程死亡：剩余文件全部显式标失败，不静默消失。
                    for remaining in ordered_indices[position:]:
                        emit_result(
                            remaining,
                            _failed_result(pending_by_index[remaining], _WORKER_DEAD_NOTE),
                        )
                    # worker 已死，来不及自删的临时文件由协调器精确清理。
                    self._discard_pending_temps()
                break

            result = self._finalize_outcome(outcome, pending, cfg)
            if result is not None:
                emit_result(index, result)
            if outcome.kind == "cancelled":
                # worker 在页边界确认取消：其余文件不再处理。
                cancelled = True
                break

        # 排空尾部事件（最后一个文件的进度/日志可能晚于 outcome 到达）。
        quiet_deadline = time.monotonic() + 0.5
        while time.monotonic() < quiet_deadline:
            if self._drain_events(progress_cb, page_start_cb, log_cb, total_files, timeout=0.05) == 0:
                break

        # 收尾兜底：清理仍挂起的临时文件与源目录中的历史残留，
        # 保证批次结束时源目录里不留 ~*.ocrtmp.pdf。
        self._discard_pending_temps()
        _cleanup_stale_temps({pending.source.parent for pending in pending_by_index.values()})

        summary = BatchSummary(
            results=_ordered_results(result_by_index),
            cancelled=cancelled,
        )
        _log_batch_end(summary)
        return summary

    # ------------------------------------------------------------------
    # worker 交互
    # ------------------------------------------------------------------

    def _warmup(
        self,
        cancel_event: Any | None,
        progress_cb: ProgressCb | None,
        page_start_cb: ProgressCb | None,
        log_cb: Callable[[str], None] | None,
        total_files: int,
    ) -> tuple[str, str]:
        """预热引擎。返回 (status, note)：ok/cancelled/failed。"""
        assert self._task_queue is not None and self._outcome_queue is not None
        started = time.monotonic()
        self._task_queue.put(WARMUP_TASK)
        while True:
            # 用户取消优先于死亡判定：取消路径的 shutdown 会终止进程，
            # 若先检查存活会把「取消」误报成「进程意外退出」。
            if _event_is_set(cancel_event):
                # 模型加载是 C++ 调用无法中断：终止进程，下次批次重建。
                self.shutdown()
                return ("cancelled", "")
            # Windows 实证：子进程死亡后父端 queue.get 永远 Empty、不会抛
            # EOFError（父进程自己持有队列 writer 句柄），死亡检测必须以
            # 进程存活状态为准。
            process = self._process
            outcome_queue = self._outcome_queue
            if process is None or not process.is_alive() or outcome_queue is None:
                return ("failed", _WORKER_DEAD_NOTE)
            self._drain_events(progress_cb, page_start_cb, log_cb, total_files, timeout=0.0)
            try:
                obj = outcome_queue.get(timeout=0.05)
            except queue_mod.Empty:
                obj = None
            except (EOFError, OSError, ValueError):
                # 队列被 shutdown 关闭：等价于进程死亡。
                return ("failed", _WORKER_DEAD_NOTE)
            if not (isinstance(obj, tuple) and len(obj) == 2):
                # 非预期的消息类型：忽略而不是解包炸掉批次线程。
                continue
            kind, note = obj
            if kind == "warmup":
                self._warmup_elapsed = time.monotonic() - started
                return ("ok", note)
            if kind == "warmup_error":
                return ("failed", note)

    def _process_one(
        self,
        task: WorkerTask,
        cancel_event: Any | None,
        progress_cb: ProgressCb | None,
        page_start_cb: ProgressCb | None,
        log_cb: Callable[[str], None] | None,
        total_files: int,
    ) -> WorkerOutcome | None:
        """提交单个文件并等待完成；None 表示取消或进程死亡（批次中止）。"""
        assert self._task_queue is not None and self._outcome_queue is not None
        self._task_queue.put(task)
        self._in_flight_temps[task.index] = Path(task.temp_output)
        soft_deadline: float | None = None

        while True:
            # shutdown 已把 _process 置 None（关窗路径）：按进程死亡收场，
            # 否则下方的队列 get 会在队列关闭后抛 ValueError 逃出批次线程。
            if self._process is None:
                self._engine_ready = False
                return None
            outcome_queue = self._outcome_queue
            if outcome_queue is None:
                self._engine_ready = False
                return None
            self._drain_events(progress_cb, page_start_cb, log_cb, total_files, timeout=0.0)
            try:
                outcome = outcome_queue.get(timeout=0.1)
            except queue_mod.Empty:
                outcome = None
            except (EOFError, OSError, ValueError):
                # 队列被 shutdown 关闭：等价于进程死亡。
                self._engine_ready = False
                return None
            if isinstance(outcome, WorkerOutcome):
                # outcome 已到：同队列 FIFO 保证该文件的页事件已全部入队，
                # 先排空再返回——否则残留事件晚于 file_done 转发，会把界面
                # 上已完成的行刷回"处理中"。
                self._drain_events(progress_cb, page_start_cb, log_cb, total_files, timeout=0.0)
                # outcome 已送达：临时文件由 worker 自删或已原子改名，无需协调器处理。
                self._in_flight_temps.pop(task.index, None)
                return outcome

            if _event_is_set(cancel_event):
                # 软取消：worker 在页边界返回 cancelled outcome；
                # 宽限超时说明卡死，硬杀进程，下次批次重建。
                if self._worker_cancel is not None:
                    self._worker_cancel.set()
                if soft_deadline is None:
                    soft_deadline = time.monotonic() + _CANCEL_SOFT_GRACE_SECONDS
                elif time.monotonic() > soft_deadline:
                    _logger.warning("worker did not honour cancel, terminating")
                    self.shutdown()
                    return None

            process = self._process
            outcome_queue = self._outcome_queue
            if process is None or outcome_queue is None or not process.is_alive():
                # 窄竞态兜底：「outcome 已入队但还没取到 + worker 随即死亡」
                # 时，最后非阻塞取一次——取到就按正常结果收尾，避免把已经
                # 成功落盘的文件误报为「工作进程意外退出」。
                try:
                    late = outcome_queue.get_nowait()
                except (queue_mod.Empty, EOFError, OSError, ValueError, AttributeError):
                    late = None
                if isinstance(late, WorkerOutcome):
                    self._drain_events(
                        progress_cb, page_start_cb, log_cb, total_files, timeout=0.0
                    )
                    self._in_flight_temps.pop(task.index, None)
                    return late
                self._engine_ready = False
                return None

    def _discard_pending_temps(self) -> None:
        """精确清理未完成任务的临时文件。

        调用时机：worker 已死亡或被终止（文件句柄已释放），此时 unlink
        安全；个别文件因句柄未完全释放而失败时留给下次批次的兜底清理。
        """
        if not self._in_flight_temps:
            return
        for temp in self._in_flight_temps.values():
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                _logger.warning("pending temp cleanup failed path=%s", temp, exc_info=True)
        self._in_flight_temps.clear()

    def _finalize_outcome(
        self,
        outcome: WorkerOutcome,
        pending: _PendingFile,
        cfg: AppConfig,
    ) -> FileResult | None:
        if outcome.kind == "ok":
            if outcome.all_existing_text or not outcome.produced_output:
                # 全部页已有文字层 / OCR 未识别出任何文字：无输出文件，
                # 也绝不触发覆盖原文件。
                status = (
                    FileStatus.SUCCESS_EXISTING_TEXT
                    if outcome.all_existing_text
                    else FileStatus.NO_TEXT_FOUND
                )
                return FileResult(
                    source=pending.source,
                    output=None,
                    status=status,
                    total_pages=outcome.total_pages,
                    ocr_pages=outcome.ocr_pages,
                    note="",
                    rel=pending.rel,
                )
            output = pending.output
            if not output.is_file():
                return _failed_result(pending, "worker returned ok without output")

            status = FileStatus.SUCCESS_OCR
            final_output = output
            note = ""
            if cfg.output.overwrite_original:
                failure = _overwrite_after_finalize(pending.source, output)
                if failure is None:
                    final_output = pending.source
                else:
                    note = failure
            return FileResult(
                source=pending.source,
                output=final_output,
                status=status,
                total_pages=outcome.total_pages,
                ocr_pages=outcome.ocr_pages,
                note=note,
                rel=pending.rel,
            )
        if outcome.kind == "encrypted":
            return FileResult(
                source=pending.source,
                output=None,
                status=FileStatus.SKIPPED_ENCRYPTED,
                total_pages=0,
                ocr_pages=0,
                note=outcome.note,
                rel=pending.rel,
            )
        if outcome.kind == "failed":
            return _failed_result(pending, outcome.note)
        if outcome.kind == "cancelled":
            return None
        return _failed_result(pending, f"未知 worker 结果: {outcome.kind}")

    # ------------------------------------------------------------------
    # 事件转发
    # ------------------------------------------------------------------

    def _drain_outcomes(self) -> None:
        """排空上一批残留的结果消息，避免旧结果被误配到本批任务。"""
        if self._outcome_queue is None:
            return
        while True:
            try:
                self._outcome_queue.get_nowait()
            except queue_mod.Empty:
                return
            except (EOFError, OSError, ValueError):
                return

    def _drain_events(
        self,
        progress_cb: ProgressCb | None,
        page_start_cb: ProgressCb | None,
        log_cb: Callable[[str], None] | None,
        total_files: int,
        *,
        timeout: float,
    ) -> int:
        """转发 worker 事件；返回本次取到的事件数。"""
        if self._events_queue is None:
            return 0
        events_queue = self._events_queue
        count = 0
        while True:
            try:
                event = events_queue.get(timeout=timeout)
            except queue_mod.Empty:
                return count
            except (EOFError, OSError, ValueError):
                # 队列被 shutdown 关闭时抛 ValueError：静默收尾即可。
                return count

            count += 1
            timeout = 0
            _handle_event(event, progress_cb, page_start_cb, log_cb, total_files)


def _failed_result(pending: _PendingFile, note: str) -> FileResult:
    return FileResult(
        source=pending.source,
        output=None,
        status=FileStatus.FAILED,
        total_pages=0,
        ocr_pages=0,
        note=note,
        rel=pending.rel,
    )


def _cleanup_stale_temps(source_dirs: set[Path]) -> None:
    removed = 0
    for source_dir in source_dirs:
        for path in source_dir.glob("~*.ocrtmp.pdf"):
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError as exc:
                _logger.warning("stale temp cleanup failed path=%s error=%s", path, exc)
    if removed:
        _logger.warning("removed stale temp files count=%d", removed)


def _handle_event(
    event: Any,
    progress_cb: ProgressCb | None,
    page_start_cb: ProgressCb | None,
    log_cb: Callable[[str], None] | None,
    total_files: int,
) -> None:
    if not isinstance(event, tuple) or not event:
        return

    if event[0] == "page" and len(event) == 5:
        if progress_cb is not None:
            _tag, index, rel, page_i, page_n = event
            # 文件序号用 worker 任务自带的 index（1 基）。
            file_i = min(int(index) + 1, total_files)
            progress_cb(file_i, total_files, int(page_i), int(page_n), str(rel))
        return

    if event[0] == "page_start" and len(event) == 5:
        if page_start_cb is not None:
            _tag, index, rel, page_i, page_n = event
            file_i = min(int(index) + 1, total_files)
            page_start_cb(file_i, total_files, int(page_i), int(page_n), str(rel))
        return

    if event[0] == "log" and len(event) >= 2:
        _log_message(str(event[1]), log_cb)


def _log_message(message: str, log_cb: Callable[[str], None] | None) -> None:
    if log_cb is not None:
        log_cb(message)
    _logger.info("%s", message)


def _ordered_results(result_by_index: dict[int, FileResult]) -> list[FileResult]:
    return [result_by_index[index] for index in sorted(result_by_index)]


def _event_is_set(event: Any | None) -> bool:
    if event is None:
        return False
    try:
        return bool(event.is_set())
    except AttributeError:
        return False
