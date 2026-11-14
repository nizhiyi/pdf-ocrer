from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from pdf_ocrer import __version__
from pdf_ocrer.app_logging import setup_logging
from pdf_ocrer.config import (
    ConfigError,
    LoggingConfig,
    OcrConfig,
    bootstrap_frozen_config,
    default_config_path,
    load_config,
)
from pdf_ocrer.ocr_engine import OcrEngineProtocol, create_engine
from pdf_ocrer.pipeline import BatchSummary, FileStatus, run_batch

EngineFactory = Callable[[OcrConfig], OcrEngineProtocol]


def _configure_utf8_output() -> None:
    """Force stdout/stderr to UTF-8 so Chinese output never crashes the CLI.

    On a Windows console whose code page is not Chinese (e.g. cp1252), printing
    the summary/log would otherwise raise UnicodeEncodeError. reconfigure() is a
    no-op-friendly best effort: guarded for streams that don't support it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


def main(
    argv: list[str] | None = None,
    *,
    engine_factory: EngineFactory | None = None,
) -> int:
    _configure_utf8_output()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.dpi is not None and not 72 <= args.dpi <= 600:
        print("设置错误: dpi 超出范围，应为 72–600", file=sys.stderr)
        return 1

    if args.version:
        print(f"pdf-ocrer {__version__}")
        return 0

    targets = [Path(path) for path in args.paths]
    if not targets:
        from pdf_ocrer.gui import run_gui

        run_gui()
        return 0

    missing = [path for path in targets if not path.exists()]
    if missing:
        for path in missing:
            print(f"路径不存在: {path}", file=sys.stderr)
        return 2

    try:
        config_path = args.config if args.config is not None else default_config_path()
        bootstrap_frozen_config(config_path)
        cfg = load_config(config_path)
        setup_logging(cfg.logging)
        if args.dpi is not None:
            cfg = replace(cfg, ocr=replace(cfg.ocr, dpi=args.dpi))
        if args.recursive:
            cfg = replace(cfg, input=replace(cfg.input, recursive=True))

        log_cb = print
        engine = (
            engine_factory(cfg.ocr)
            if engine_factory is not None
            else _create_default_engine(cfg.ocr, log_cb)
        )
        summary = run_batch(
            targets,
            cfg,
            engine,
            progress_cb=_print_progress,
            log_cb=log_cb,
            force=args.force,
        )
    except ConfigError as exc:
        setup_logging(LoggingConfig())
        logging.getLogger("pdf_ocrer").error("设置错误: %s", exc)
        print(f"设置错误: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        setup_logging(LoggingConfig())
        logging.getLogger("pdf_ocrer").error("运行失败: %s", exc)
        print(f"运行失败: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary)
    if not summary.results:
        return 2
    if any(result.status is FileStatus.FAILED for result in summary.results):
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pdf-ocrer")
    parser.add_argument("paths", nargs="*", help="要处理的 PDF/图片文件或文件夹，可传多个")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--recursive", action="store_true", help="递归扫描子文件夹")
    parser.add_argument("--force", action="store_true", help="输出文件已存在时也重新处理")
    parser.add_argument("--version", action="store_true")
    return parser


def _create_default_engine(
    cfg: OcrConfig,
    log: Callable[[str], None] | None = None,
) -> OcrEngineProtocol:
    return create_engine(cfg, log)


def _print_progress(file_i: int, file_n: int, page_i: int, page_n: int, filename: str) -> None:
    print(f"[{file_i}/{file_n}] {filename} 第 {page_i}/{page_n} 页", flush=True)


def _print_summary(summary: BatchSummary) -> None:
    print("原文件名\t输出文件\t状态\t总页数\tOCR页数\t备注")
    for result in summary.results:
        print(
            "\t".join(
                [
                    result.rel or result.source.name,
                    "" if result.output is None else result.output.name,
                    result.status.value,
                    str(result.total_pages),
                    str(result.ocr_pages),
                    result.note,
                ]
            )
        )
