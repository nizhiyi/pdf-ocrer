from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pdf_ocrer.app_logging import setup_logging
from pdf_ocrer.config import LoggingConfig


@pytest.fixture(autouse=True)
def clean_pdf_ocrer_file_handlers():
    logger = logging.getLogger("pdf_ocrer")
    original_level = logger.level
    _remove_pdf_ocrer_file_handlers(logger)
    try:
        yield
    finally:
        _remove_pdf_ocrer_file_handlers(logger)
        logger.setLevel(original_level)


def test_setup_logging_writes_file_with_level_and_message(tmp_path: Path) -> None:
    log_path = setup_logging(LoggingConfig(level="DEBUG", dir=str(tmp_path)))

    logging.getLogger("pdf_ocrer").debug("hello file log")
    _flush_pdf_ocrer_file_handlers()

    assert log_path == tmp_path / "pdf_ocrer.log"
    text = (tmp_path / "pdf_ocrer.log").read_text(encoding="utf-8")
    assert "DEBUG pdf_ocrer: hello file log" in text


def test_setup_logging_filters_debug_at_info_level(tmp_path: Path) -> None:
    setup_logging(LoggingConfig(level="INFO", dir=str(tmp_path)))

    logging.getLogger("pdf_ocrer").debug("hidden debug")
    logging.getLogger("pdf_ocrer").info("visible info")
    _flush_pdf_ocrer_file_handlers()

    text = (tmp_path / "pdf_ocrer.log").read_text(encoding="utf-8")
    assert "hidden debug" not in text
    assert "INFO pdf_ocrer: visible info" in text


def test_setup_logging_is_idempotent_for_same_target(tmp_path: Path) -> None:
    first = setup_logging(LoggingConfig(dir=str(tmp_path)))
    second = setup_logging(LoggingConfig(dir=str(tmp_path)))

    handlers = [
        handler
        for handler in logging.getLogger("pdf_ocrer").handlers
        if getattr(handler, "_pdf_ocrer_file_handler", False)
    ]
    assert second == first
    assert len(handlers) == 1


def _remove_pdf_ocrer_file_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_pdf_ocrer_file_handler", False):
            logger.removeHandler(handler)
            handler.close()


def _flush_pdf_ocrer_file_handlers() -> None:
    for handler in logging.getLogger("pdf_ocrer").handlers:
        if getattr(handler, "_pdf_ocrer_file_handler", False):
            handler.flush()
