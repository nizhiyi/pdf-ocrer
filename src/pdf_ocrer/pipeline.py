from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from uuid import uuid4

from pdf_ocrer.config import AppConfig, InputConfig
from pdf_ocrer.ocr_engine import OcrEngineProtocol
from pdf_ocrer.pdf_processor import BatchCancelled, EncryptedPdfError, PdfResult, process_pdf
from pdf_ocrer.scanning import (
    OUTPUT_SUFFIX,
    ScanItem,
    allowed_extensions,
    is_generated_output,
    scan_inputs,
)


class FileStatus(str, Enum):
    SUCCESS_OCR = "OCR完成"
    SUCCESS_EXISTING_TEXT = "已OCR-跳过"
    NO_TEXT_FOUND = "无文本-跳过"
    SKIPPED_ENCRYPTED = "加密-跳过"
    SKIPPED_DONE = "已输出-跳过"
    FAILED = "失败"


@dataclass
class FileResult:
    source: Path
    output: Path | None
    status: FileStatus
    total_pages: int
    ocr_pages: int
    note: str
    rel: str = ""


@dataclass
class BatchSummary:
    results: list[FileResult]
    cancelled: bool


ProgressCb = Callable[[int, int, int, int, str], None]

_logger = logging.getLogger(__name__)

_SLOW_MODEL_MESSAGE = (
    "提示：当前使用 PP-OCR {version} {size} 模型，CPU 上单页明显更慢。"
    "若需更快，请在「设置」中把模型大小改为 small（v6）或 mobile（v4/v5）。"
)


def slow_model_warning(cfg: AppConfig) -> str | None:
    """当所选模型尺寸明显很慢时返回提示文案，否则 None。

    medium（v6）/server（v4/v5）在 CPU 上单页明显更慢，普通用户很容易
    误以为是"卡在加载模型"。
    """
    size = cfg.ocr.model_type.casefold()
    if size in {"medium", "server"}:
        return _SLOW_MODEL_MESSAGE.format(version=cfg.ocr.ocr_version, size=size)
    return None


def emit_slow_model_warning(cfg: AppConfig, log_cb: Callable[[str], None] | None) -> None:
    warning = slow_model_warning(cfg)
    if warning is None:
        return
    if log_cb is not None:
        log_cb(warning)
    _logger.warning("%s", warning)


def expand_targets(targets: list[Path], input_cfg: InputConfig) -> list[ScanItem]:
    """展开混合的文件/文件夹输入为 ScanItem 列表。

    - 文件：直接成为一项（rel 为文件名）；扩展名不在允许列表的记录日志后忽略。
    - 文件夹：按 cfg 递归/非递归扫描。
    - 按绝对路径去重（大小写不敏感），rel 冲突时逐级加父目录名前缀消歧，
      保证 rel 全局唯一（GUI 表格行键与进度显示依赖它）。
    """
    items: list[ScanItem] = []
    for target in targets:
        target = Path(target)
        if target.is_dir():
            items.extend(scan_inputs(target, input_cfg))
        elif target.is_file():
            if target.suffix.removeprefix(".").casefold() not in allowed_extensions(input_cfg):
                _logger.info("skipped unsupported input file=%s", target)
                continue
            # 本工具的输出文件（*_OCR.pdf）绝不作为输入：既无意义，覆盖模式
            # 下还会把上次的输出改写掉（表现为"_OCR.pdf 被删除"）。
            if is_generated_output(target):
                _logger.info("skipped generated output file=%s", target)
                continue
            items.append(ScanItem(src=target, rel=target.name))
        else:
            # 不存在的显式路径：保留为条目，处理阶段按 FileNotFoundError 失败，
            # 在结果表中给出明确的失败行而不是静默消失。
            items.append(ScanItem(src=target, rel=target.name))

    return _dedupe_and_disambiguate(items)


def _dedupe_and_disambiguate(items: list[ScanItem]) -> list[ScanItem]:
    seen_paths: set[str] = set()
    unique: list[ScanItem] = []
    for item in items:
        key = str(item.src).casefold()
        if key in seen_paths:
            continue
        seen_paths.add(key)
        unique.append(item)

    # 按原始 rel 分组：组内多于一个成员（跨源同名）时，组内全部成员
    # 逐级加父目录名前缀消歧，保持命名一致；单成员组保持原 rel。
    groups: dict[str, list[ScanItem]] = {}
    for item in unique:
        groups.setdefault(item.rel.casefold(), []).append(item)

    used_rels: set[str] = set()
    result: list[ScanItem] = []
    for members in groups.values():
        if len(members) == 1:
            item = members[0]
            result.append(ScanItem(src=item.src, rel=item.rel))
            used_rels.add(item.rel.casefold())
            continue
        for src, rel in _disambiguate_group(members, used_rels):
            result.append(ScanItem(src=src, rel=rel))
            used_rels.add(rel.casefold())

    result.sort(key=lambda item: item.rel.casefold())
    return result


def _disambiguate_group(
    members: list[ScanItem],
    used_rels: set[str],
) -> list[tuple[Path, str]]:
    taken = set(used_rels)
    assigned: list[tuple[Path, str]] = []
    for item in members:
        max_depth = max(1, len(item.src.parts) - len(PurePosixPath(item.rel).parts))
        candidate = item.rel
        for depth in range(1, max_depth + 1):
            candidate = _rel_with_parents(item.src, item.rel, depth)
            if candidate.casefold() not in taken:
                break
        assigned.append((item.src, candidate))
        taken.add(candidate.casefold())
    return assigned


def _rel_with_parents(src: Path, rel: str, depth: int) -> str:
    rel_parts = PurePosixPath(rel).parts
    rel_len = len(rel_parts)
    start = max(0, len(src.parts) - rel_len - depth)
    prefix = list(src.parts[start:len(src.parts) - rel_len])
    return "/".join([*prefix, *rel_parts])


def run_batch(
    targets: list[Path],
    cfg: AppConfig,
    engine: OcrEngineProtocol,
    progress_cb: ProgressCb | None = None,
    log_cb: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    file_cb: Callable[[FileResult], None] | None = None,
    pending_cb: Callable[[list[str]], None] | None = None,
    page_start_cb: ProgressCb | None = None,
    force: bool = False,
) -> BatchSummary:
    """批量处理文件/文件夹混合输入，输出 ``原文件名_OCR.pdf`` 到源文件同目录。

    默认当目标输出文件已存在时跳过该文件（状态"已输出-跳过"）；
    ``force=True`` 时无视已存在的输出，重新处理并覆盖。
    """
    items = expand_targets(targets, cfg.input)
    _logger.info("batch start targets=%s files=%d", [str(path) for path in targets], len(items))
    # 先把全部待处理文件告诉界面，让表格在开始处理时就列出完整清单。
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

    results: list[FileResult] = []
    cancelled = False

    for file_i, item in enumerate(items, start=1):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break

        src = item.src
        # 处理间隙被移动/删除的文件：显式归失败并记录原因，不中断批次。
        if not src.is_file():
            result = FileResult(
                source=src,
                output=None,
                status=FileStatus.FAILED,
                total_pages=0,
                ocr_pages=0,
                note="文件不存在或已被移动",
                rel=item.rel,
            )
            results.append(result)
            _log_file_result(result)
            message = _file_log_message(result)
            if message is not None and log_cb is not None:
                log_cb(message)
            if file_cb is not None:
                file_cb(result)
            continue

        output = src.parent / f"{src.stem}{OUTPUT_SUFFIX}.pdf"
        if _should_skip_existing_output(src, output, cfg, force):
            result = FileResult(
                source=src,
                output=output,
                status=FileStatus.SKIPPED_DONE,
                total_pages=0,
                ocr_pages=0,
                note="",
                rel=item.rel,
            )
            results.append(result)
            _log_file_result(result)
            message = f"跳过（已输出）: {item.rel}"
            if log_cb is not None:
                log_cb(message)
            _logger.info("%s", message)
            if file_cb is not None:
                file_cb(result)
            continue

        try:
            processed = process_pdf(
                src,
                cfg,
                engine,
                page_cb=_page_progress(progress_cb, file_i, len(items), item.rel),
                cancel=cancel_event,
                page_start_cb=_page_progress(
                    page_start_cb, file_i, len(items), item.rel
                ),
            )
            result = _finalize_processed_file(src, item.rel, processed, cfg)
        except EncryptedPdfError as exc:
            result = FileResult(
                source=src,
                output=None,
                status=FileStatus.SKIPPED_ENCRYPTED,
                total_pages=0,
                ocr_pages=0,
                note=str(exc),
                rel=item.rel,
            )
        except BatchCancelled:
            cancelled = True
            break
        # OCR 引擎与 PDF 栈可抛任意异常：按文件隔离为 FAILED，
        # 保证批量任务中单个坏文件不中断其余文件的处理。
        except Exception as exc:  # noqa: BLE001
            result = FileResult(
                source=src,
                output=None,
                status=FileStatus.FAILED,
                total_pages=0,
                ocr_pages=0,
                note=_exception_note(exc),
                rel=item.rel,
            )

        results.append(result)
        _log_file_result(result)
        message = _file_log_message(result)
        if message is not None and log_cb is not None:
            log_cb(message)
        if file_cb is not None:
            file_cb(result)

    summary = BatchSummary(results=results, cancelled=cancelled)
    _log_batch_end(summary)
    return summary


def _file_log_message(result: FileResult) -> str | None:
    """单个文件终态的用户日志行；跳过状态由调用方控制输出时序，返回 None。"""
    if result.status is FileStatus.SKIPPED_DONE:
        return None
    rel = result.rel
    if result.status is FileStatus.FAILED:
        return f"失败: {rel}（{result.note}）" if result.note else f"失败: {rel}"
    if result.status is FileStatus.SKIPPED_ENCRYPTED:
        return f"跳过（加密）: {rel}（{result.note}）" if result.note else f"跳过（加密）: {rel}"
    if result.status is FileStatus.SUCCESS_EXISTING_TEXT:
        return f"跳过（已OCR）: {rel}"
    if result.status is FileStatus.NO_TEXT_FOUND:
        return f"跳过（无文本）: {rel}"
    if result.status is FileStatus.SUCCESS_OCR:
        message = f"OCR 完成: {rel}（OCR {result.ocr_pages}/{result.total_pages} 页）"
        if result.note:
            # 覆盖模式替换失败等场景：OCR 本身成功但有需要用户知道的问题。
            message += f"，{result.note}"
        return message
    return None


def _should_skip_existing_output(src: Path, output: Path, cfg: AppConfig, force: bool) -> bool:
    """输出已存在时是否跳过。

    覆盖模式对 PDF 重跑并替换原文件，因此不做存在性跳过；图片永远不能
    覆盖原文件（只能导出 _OCR），输出已存在就该跳过，避免重复 OCR。
    """
    if force or not output.exists():
        return False
    return not (cfg.output.overwrite_original and src.suffix.casefold() == ".pdf")


def _page_progress(
    progress_cb: ProgressCb | None,
    file_i: int,
    file_n: int,
    filename: str,
) -> Callable[[int, int], None] | None:
    if progress_cb is None:
        return None

    def callback(page_i: int, page_n: int) -> None:
        progress_cb(file_i, file_n, page_i + 1, page_n, filename)

    return callback


def _finalize_processed_file(
    src: Path,
    rel: str,
    processed: PdfResult,
    cfg: AppConfig,
) -> FileResult:
    doc_closed = False
    try:
        # 空文档（0 页）reports 为空：all() 恒 True 会误报「已有文字层」。
        all_existing_text = bool(processed.reports) and all(
            report.action == "kept_existing" for report in processed.reports
        )
        # OCR 没有在任何页面写出文字（全部页 empty）：与「已有文字层」一样
        # 不产出输出——生成一份没有文字层的拷贝毫无意义。
        has_ocr_pages = any(report.action == "ocr" for report in processed.reports)

        if all_existing_text:
            # 全部页已含文字层：无需 OCR，不产出任何输出文件。
            processed.doc.close()
            doc_closed = True
            return FileResult(
                source=src,
                output=None,
                status=FileStatus.SUCCESS_EXISTING_TEXT,
                total_pages=processed.total_pages,
                ocr_pages=processed.ocr_pages,
                note="",
                rel=rel,
            )
        if not has_ocr_pages:
            # 无文字：跳过该文件，不产出输出、不覆盖原文件。
            processed.doc.close()
            doc_closed = True
            return FileResult(
                source=src,
                output=None,
                status=FileStatus.NO_TEXT_FOUND,
                total_pages=processed.total_pages,
                ocr_pages=processed.ocr_pages,
                note="",
                rel=rel,
            )

        # 输出路径由源文件确定性决定（原文件名_OCR.pdf，与源同目录）；
        # force/覆盖模式重跑时直接覆盖旧输出。
        output = src.parent / f"{src.stem}{OUTPUT_SUFFIX}.pdf"
        processed.doc.subset_fonts()
        # 同目录临时文件 + 原子改名（os.replace）：save 中途失败最多留下
        # 可清理的 ~*.ocrtmp.pdf 垃圾，最终 _OCR.pdf 要么不存在、要么完整。
        temp_output = output.parent / f"~{uuid4().hex}.ocrtmp.pdf"
        try:
            processed.doc.save(temp_output, garbage=3, deflate=True)
            os.replace(temp_output, output)
        except BaseException:
            temp_output.unlink(missing_ok=True)
            raise
        processed.doc.close()
        doc_closed = True
        # 无 ocr 页的文件在上方已提前返回：走到这里的输出必然含 OCR 文字层。
        status = FileStatus.SUCCESS_OCR

        final_output = output
        note = ""
        if cfg.output.overwrite_original:
            failure = _overwrite_after_finalize(src, output)
            if failure is None:
                final_output = src
            else:
                note = failure

        return FileResult(
            source=src,
            output=final_output,
            status=status,
            total_pages=processed.total_pages,
            ocr_pages=processed.ocr_pages,
            note=note,
            rel=rel,
        )
    finally:
        if not doc_closed:
            processed.doc.close()


def _overwrite_after_finalize(src: Path, output: Path) -> str | None:
    """覆盖模式后处理：备份原文件，再用 _OCR 结果替换原文件。

    仅在 `_OCR` 已成功生成后调用；任何失败都保留 `_OCR` 文件并返回原因
    （None 表示成功）。图片输入不能覆盖（扩展名会与内容不一致），退化为
    导出 `_OCR` 文件。本工具的输出文件（*_OCR.pdf）也绝不覆盖：它们是
    产物而非源文件，覆盖会改写/消耗上次的输出（表现为"_OCR 被删除"）。
    """
    if src.suffix.casefold() != ".pdf":
        return "图片输入不支持覆盖原文件，已保留 _OCR 文件"
    if is_generated_output(src):
        _logger.warning("refusing to overwrite generated output src=%s", src)
        return "源文件是本工具的输出文件（*_OCR.pdf），已跳过覆盖并保留 _OCR 文件"

    from pdf_ocrer.backup import backup_original

    try:
        backup_original(src)
        os.replace(output, src)
        return None
    except OSError as exc:
        reason = _exception_note(exc)
        _logger.warning("overwrite original failed src=%s error=%s", src, reason)
        return f"覆盖原文件失败，已保留 _OCR 文件：{reason}"



def _log_file_result(result: FileResult) -> None:
    source_name = result.rel or result.source.name
    output_name = "" if result.output is None else result.output.name
    _logger.info(
        "file result source=%s status=%s output=%s note=%s",
        source_name,
        result.status.value,
        output_name,
        result.note,
    )


def _exception_note(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _log_batch_end(summary: BatchSummary) -> None:
    failed = sum(result.status is FileStatus.FAILED for result in summary.results)
    skipped = sum(result.status is FileStatus.SKIPPED_ENCRYPTED for result in summary.results)
    skipped_done = sum(result.status is FileStatus.SKIPPED_DONE for result in summary.results)
    existing = sum(result.status is FileStatus.SUCCESS_EXISTING_TEXT for result in summary.results)
    no_text = sum(result.status is FileStatus.NO_TEXT_FOUND for result in summary.results)
    _logger.info(
        "batch end results=%d failed=%d skipped=%d skipped_done=%d existing=%d no_text=%d"
        " cancelled=%s",
        len(summary.results),
        failed,
        skipped,
        skipped_done,
        existing,
        no_text,
        summary.cancelled,
    )
