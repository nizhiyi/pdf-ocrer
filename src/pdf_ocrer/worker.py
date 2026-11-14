"""Spawn-safe worker boundary for OCR processing in a separate process.

This module is imported inside the Windows ``spawn`` worker process, so every
callable and data object used across the process boundary lives at module
scope. The pickle boundary is intentionally narrow: the worker receives
``WorkerTask`` objects and returns ``WorkerOutcome`` objects. PyMuPDF
documents, rendered images, and OCR engine instances never cross process
boundaries.

Worker events are plain tuples. ``("page", index, rel, page_i, page_n)``
drives progress in the coordinator. ``("log", msg)`` carries engine log
messages (e.g. model loading) to the coordinator's log callback.

The worker process is long-lived: it is started once per app run and serves
file tasks sequentially until it receives the ``None`` sentinel, so the OCR
model is loaded exactly once instead of once per batch.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue as queue_mod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pdf_ocrer.config import AppConfig, DebugConfig, OcrConfig, OutputConfig
from pdf_ocrer.ocr_engine import OcrEngineProtocol, create_engine
from pdf_ocrer.pdf_processor import BatchCancelled, EncryptedPdfError, process_pdf


@dataclass(frozen=True)
class WorkerTask:
    index: int
    source: str
    rel: str
    output: str
    # 同目录临时文件路径：由协调器生成并持有，便于任何异常路径下
    # 精确清理（worker 只负责 save → 原子改名 → 失败时自删）。
    temp_output: str


@dataclass(frozen=True)
class WorkerOutcome:
    index: int
    rel: str
    kind: str
    total_pages: int
    ocr_pages: int
    all_existing_text: bool
    # False 表示该文件不产出输出（全部页已有文字层 / OCR 未识别出任何文字），
    # 状态由协调器按 all_existing_text 区分「已OCR-跳过」与「无文本-跳过」。
    produced_output: bool
    note: str


_CANCEL: Any | None = None
_EVENTS: Any | None = None
_OCR_CFG = OcrConfig()
_DEBUG_CFG = DebugConfig()
_CFG = AppConfig(_OCR_CFG, OutputConfig(), _DEBUG_CFG)
_ENGINE: OcrEngineProtocol | None = None
_LOG_HANDLER: logging.Handler | None = None
_engine_factory: Callable[..., OcrEngineProtocol] = create_engine

# 父进程存活看门的轮询间隔（秒）：任务队列 get 的超时即轮询节拍。
_WORKER_POLL_SECONDS = 1.0

WARMUP_TASK = "warmup"
"""任务队列哨兵：预热引擎。``None`` 表示退出。"""


class _EventQueueHandler(logging.Handler):
    def __init__(self, events: Any) -> None:
        super().__init__(level=logging.INFO)
        self._events = events

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._events.put(("log", record.getMessage()))
        # 跨进程队列失效可抛任意异常；走 logging 标准容错路径而非静默丢弃
        except Exception:  # noqa: BLE001
            self.handleError(record)


def init_worker(ocr_cfg: OcrConfig, debug_cfg: DebugConfig, cancel: Any, events: Any) -> None:
    global _CANCEL, _EVENTS, _OCR_CFG, _DEBUG_CFG, _CFG, _ENGINE

    _CANCEL = cancel
    _EVENTS = events
    _OCR_CFG = ocr_cfg
    _DEBUG_CFG = debug_cfg
    _CFG = AppConfig(ocr_cfg, OutputConfig(), debug_cfg)
    _ENGINE = None
    _configure_event_logging(events)


def _configure_event_logging(events: Any) -> None:
    global _LOG_HANDLER

    logger = logging.getLogger("pdf_ocrer")
    if _LOG_HANDLER is not None:
        logger.removeHandler(_LOG_HANDLER)
        _LOG_HANDLER.close()
        _LOG_HANDLER = None

    if events is None:
        return

    handler = _EventQueueHandler(events)
    logger.addHandler(handler)
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    _LOG_HANDLER = handler


def warmup() -> str:
    engine = _get_engine()
    engine.recognize(np.full((8, 8, 3), 255, dtype=np.uint8))
    return f"{type(engine).__module__}.{type(engine).__qualname__}"


def worker_serve(
    task_queue: Any,
    outcome_queue: Any,
    events_queue: Any,
    cancel: Any,
    ocr_cfg: OcrConfig,
    debug_cfg: DebugConfig,
) -> None:
    """常驻工作子进程主循环：顺序处理任务队列，直到收到退出哨兵。

    - ``"warmup"``：预热引擎（加载模型），结果以 ``("warmup", ...)`` 返回；
    - ``WorkerTask``：处理单个文件，返回 ``WorkerOutcome``；
    - ``None``：退出。

    父进程死亡时自动退出（watchdog），避免留下占内存的孤儿进程。
    """
    init_worker(ocr_cfg, debug_cfg, cancel, events_queue)

    while True:
        parent = multiprocessing.parent_process()
        if parent is not None and not parent.is_alive():
            return
        try:
            task = task_queue.get(timeout=_WORKER_POLL_SECONDS)
        except queue_mod.Empty:
            continue
        except (EOFError, OSError):
            return

        if task is None:
            return
        if task == WARMUP_TASK:
            outcome_queue.put(_serve_warmup())
            continue
        try:
            outcome_queue.put(process_file_task(task))
        except (EOFError, OSError):
            return


def _serve_warmup() -> tuple[str, str]:
    """预热引擎；加载失败的原因必须送达协调器，转为每个文件的失败结果。"""
    try:
        engine_name = warmup()
        return ("warmup", engine_name)
    # 模型下载/加载/推理栈可抛任意异常，兜底转为可读原因
    except Exception as exc:  # noqa: BLE001
        return ("warmup_error", f"{type(exc).__name__}: {exc}")


def process_file_task(task: WorkerTask) -> WorkerOutcome:
    source = Path(task.source)
    output = Path(task.output)
    # 协调器预生成的同目录临时文件：save 中途失败或进程被杀，最多留下
    # 可辨识、可清理的 ~*.ocrtmp.pdf 垃圾；最终 _OCR.pdf 经原子改名落盘，
    # 要么不存在、要么完整，绝不会留下被"输出已存在"检查永久跳过的半截成品。
    temp_output = Path(task.temp_output)

    # 排队后到执行前被移动/删除的文件：直接归失败，不触碰 OCR 引擎。
    if not source.is_file():
        return _error_outcome(task, "failed", "文件不存在或已被移动")

    try:
        engine = _get_engine()
        processed = process_pdf(
            source,
            _CFG,
            engine,
            page_cb=_page_event_callback(task),
            cancel=_CANCEL,
            page_start_cb=_page_start_event_callback(task),
        )
        doc_closed = False
        try:
            # 空文档（0 页）reports 为空：all() 恒 True 会误报「已有文字层」，
            # 必须先确认确实存在页记录。
            all_existing_text = bool(processed.reports) and all(
                report.action == "kept_existing" for report in processed.reports
            )
            # OCR 没有在任何页面写出文字（全部页 empty）：与「已有文字层」
            # 一样不产出输出——生成一份没有文字层的拷贝毫无意义。
            produced_output = any(report.action == "ocr" for report in processed.reports)

            if all_existing_text or not produced_output:
                # 不产出输出文件，也绝不触发覆盖原文件。
                processed.doc.close()
                doc_closed = True
                return WorkerOutcome(
                    index=task.index,
                    rel=task.rel,
                    kind="ok",
                    total_pages=processed.total_pages,
                    ocr_pages=processed.ocr_pages,
                    all_existing_text=all_existing_text,
                    produced_output=False,
                    note="",
                )

            output.parent.mkdir(parents=True, exist_ok=True)
            processed.doc.subset_fonts()
            processed.doc.save(temp_output, garbage=3, deflate=True)
            os.replace(temp_output, output)
            processed.doc.close()
            doc_closed = True

            return WorkerOutcome(
                index=task.index,
                rel=task.rel,
                kind="ok",
                total_pages=processed.total_pages,
                ocr_pages=processed.ocr_pages,
                all_existing_text=False,
                produced_output=True,
                note="",
            )
        finally:
            if not doc_closed:
                processed.doc.close()
    except EncryptedPdfError as exc:
        _discard_partial_output(temp_output)
        return _error_outcome(task, "encrypted", str(exc))
    except BatchCancelled:
        _discard_partial_output(temp_output)
        return _error_outcome(task, "cancelled", "")
    # OCR/渲染栈可抛任意异常：兜底转为带原因的失败结果，
    # 子进程绝不能因未捕获异常崩溃（否则父进程只见进程死亡）。
    except Exception as exc:  # noqa: BLE001
        _discard_partial_output(temp_output)
        return _error_outcome(task, "failed", f"{type(exc).__name__}: {exc}")


def _discard_partial_output(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _get_engine() -> OcrEngineProtocol:
    global _ENGINE

    if _ENGINE is None:
        _ENGINE = _create_engine()
    return _ENGINE


def _create_engine() -> OcrEngineProtocol:
    try:
        return _engine_factory(_OCR_CFG, _log_event)
    except TypeError as two_arg_error:
        try:
            return _engine_factory(_OCR_CFG)
        except TypeError:
            try:
                return _engine_factory()
            except TypeError:
                raise two_arg_error


def _page_event_callback(task: WorkerTask) -> Callable[[int, int], None]:
    def callback(page_index: int, total_pages: int) -> None:
        _put_event(("page", task.index, task.rel, page_index + 1, total_pages))

    return callback


def _page_start_event_callback(task: WorkerTask) -> Callable[[int, int], None]:
    def callback(page_index: int, total_pages: int) -> None:
        _put_event(("page_start", task.index, task.rel, page_index + 1, total_pages))

    return callback


def _log_event(message: str) -> None:
    _put_event(("log", message))


def _put_event(event: tuple[Any, ...]) -> None:
    if _EVENTS is None:
        return

    try:
        _EVENTS.put(event)
    # 事件投递的最后防线：队列在进程退出竞态下可能抛任意异常，
    # 此处只能静默放弃，且不能再触发日志（emit 已自身容错）。
    except Exception:  # noqa: S110, BLE001
        pass


def _error_outcome(task: WorkerTask, kind: str, note: str) -> WorkerOutcome:
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
