from __future__ import annotations

import logging
import sys
import tomllib
import types
from pathlib import Path

import pymupdf
import pytest
from fixtures_gen import GT_LINES

from pdf_ocrer import __version__
from pdf_ocrer.cli import main
from pdf_ocrer.config import ConfigError
from pdf_ocrer.ocr_engine import OcrLine
from pdf_ocrer.pipeline import BatchSummary, FileResult, FileStatus

_DPI = 200
_FONT = pymupdf.Font("cjk")


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # CLI 不带 --config 时读取 cwd 的 config.toml；切到临时目录，
    # 避免读到开发/安装目录里的真实配置（如 overwrite_original = true）。
    monkeypatch.chdir(tmp_path)


class FakeEngine:
    def recognize(self, img_rgb) -> list[OcrLine]:
        return _gt_ocr_lines()


def test_main_without_paths_launches_gui(monkeypatch) -> None:
    called: list[bool] = []
    fake_gui = types.ModuleType("pdf_ocrer.gui")
    fake_gui.run_gui = lambda: called.append(True)
    monkeypatch.setitem(sys.modules, "pdf_ocrer.gui", fake_gui)

    assert main([]) == 0
    assert called == [True]


def test_main_version(capsys) -> None:
    assert main(["--version"]) == 0

    captured = capsys.readouterr()
    assert __version__ in captured.out


def test_main_missing_path_returns_2(tmp_path) -> None:
    assert main([str(tmp_path / "missing")]) == 2


def test_main_folder_creates_outputs_and_passes_dpi(work_folder, capsys) -> None:
    _keep_only(work_folder, {"scanned.pdf", "native.pdf"})
    captured: dict[str, object] = []

    def engine_factory(ocr_cfg):
        captured.append(ocr_cfg.dpi)
        return FakeEngine()

    exit_code = main(
        [str(work_folder), "--dpi", "300"],
        engine_factory=engine_factory,
    )

    assert exit_code == 0
    assert captured == [300]
    # 全部页已含文字层的 native.pdf 不产出文件；scanned.pdf 正常输出。
    assert not (work_folder / "native_OCR.pdf").exists()
    assert (work_folder / "scanned_OCR.pdf").exists()
    stdout = capsys.readouterr().out
    assert "native.pdf" in stdout
    assert "scanned_OCR.pdf" in stdout
    assert "OCR完成" in stdout
    assert FileStatus.SUCCESS_EXISTING_TEXT.value in stdout


def test_main_explicit_single_file_processes_to_same_dir(work_folder, capsys) -> None:
    _keep_only(work_folder, {"scanned.pdf"})

    exit_code = main([str(work_folder / "scanned.pdf")], engine_factory=lambda ocr_cfg: FakeEngine())

    assert exit_code == 0
    assert (work_folder / "scanned_OCR.pdf").exists()
    stdout = capsys.readouterr().out
    assert FileStatus.SUCCESS_OCR.value in stdout


def test_main_writes_batch_log_from_config_dir(work_folder, tmp_path) -> None:
    _keep_only(work_folder, {"scanned.pdf", "native.pdf"})
    log_dir = tmp_path / "logs"
    log_dir_text = str(log_dir).replace("\\", "/")
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'[logging]\ndir = "{log_dir_text}"\n', encoding="utf-8")

    exit_code = main(
        [str(work_folder), "--config", str(config_path)],
        engine_factory=lambda ocr_cfg: FakeEngine(),
    )

    _flush_pdf_ocrer_file_handlers()
    text = (log_dir / "pdf_ocrer.log").read_text(encoding="utf-8")
    assert exit_code == 0
    assert "batch start" in text
    assert "source=native.pdf" in text
    assert "source=scanned.pdf" in text
    assert "batch end" in text


def test_main_config_reaches_engine_factory(work_folder, tmp_path) -> None:
    _keep_only(work_folder, {"scanned.pdf", "native.pdf"})
    config_path = tmp_path / "config.toml"
    config_path.write_text('[ocr]\nocr_version = "PP-OCRv5"\nmodel_type = "server"\n', encoding="utf-8")
    captured: dict[str, object] = {}

    def engine_factory(ocr_cfg):
        captured["ocr_version"] = ocr_cfg.ocr_version
        return FakeEngine()

    exit_code = main(
        [str(work_folder), "--config", str(config_path)],
        engine_factory=engine_factory,
    )

    assert exit_code == 0
    assert captured == {"ocr_version": "PP-OCRv5"}


def test_main_rejects_removed_workers_flag(tmp_path) -> None:
    # 0.9 起已删除「同时处理文件数」（--workers）：传该参数应直接报参数错误。
    folder = tmp_path / "input"
    folder.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        main([str(folder), "--workers", "2"])

    assert excinfo.value.code == 2


def test_main_recursive_flag_overrides_input_config(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "input"
    folder.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text("[input]\nrecursive = false\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_batch(targets, cfg, engine, **kwargs):
        captured["recursive"] = cfg.input.recursive
        captured["targets"] = targets
        return BatchSummary(
            results=[
                FileResult(
                    source=targets[0] / "a.pdf",
                    output=targets[0] / "a_OCR.pdf",
                    status=FileStatus.SUCCESS_OCR,
                    total_pages=1,
                    ocr_pages=1,
                    note="",
                    rel="a.pdf",
                )
            ],
            cancelled=False,
        )

    monkeypatch.setattr("pdf_ocrer.cli.run_batch", fake_run_batch)

    exit_code = main(
        [str(folder), "--config", str(config_path), "--recursive"],
        engine_factory=lambda ocr_cfg: FakeEngine(),
    )

    assert exit_code == 0
    assert captured == {"targets": [folder], "recursive": True}


def test_main_force_flag_passes_to_run_batch(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "input"
    folder.mkdir()
    captured: dict[str, object] = {}

    def fake_run_batch(targets, cfg, engine, **kwargs):
        captured["force"] = kwargs.get("force")
        return BatchSummary(
            results=[
                FileResult(
                    source=targets[0] / "a.pdf",
                    output=targets[0] / "a_OCR.pdf",
                    status=FileStatus.SUCCESS_OCR,
                    total_pages=1,
                    ocr_pages=1,
                    note="",
                    rel="a.pdf",
                )
            ],
            cancelled=False,
        )

    monkeypatch.setattr("pdf_ocrer.cli.run_batch", fake_run_batch)

    exit_code = main(
        [str(folder), "--force"],
        engine_factory=lambda ocr_cfg: FakeEngine(),
    )

    assert exit_code == 0
    assert captured == {"force": True}


def test_main_multiple_paths_processed_in_one_batch(tmp_path, monkeypatch) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    captured: dict[str, object] = {}

    def fake_run_batch(targets, cfg, engine, **kwargs):
        captured["targets"] = targets
        return BatchSummary(
            results=[
                FileResult(
                    source=targets[0] / "a.pdf",
                    output=targets[0] / "a_OCR.pdf",
                    status=FileStatus.SUCCESS_OCR,
                    total_pages=1,
                    ocr_pages=1,
                    note="",
                    rel="a.pdf",
                )
            ],
            cancelled=False,
        )

    monkeypatch.setattr("pdf_ocrer.cli.run_batch", fake_run_batch)

    exit_code = main(
        [str(first), str(second)],
        engine_factory=lambda ocr_cfg: FakeEngine(),
    )

    assert exit_code == 0
    assert captured == {"targets": [first, second]}


def test_main_all_skipped_returns_zero(work_folder, capsys) -> None:
    _keep_only(work_folder, {"scanned.pdf"})

    first = main([str(work_folder)], engine_factory=lambda ocr_cfg: FakeEngine())
    assert first == 0

    # 第二轮：输出已存在，全部跳过仍返回 0。
    second = main([str(work_folder)], engine_factory=lambda ocr_cfg: FakeEngine())

    stdout = capsys.readouterr().out
    assert second == 0
    assert FileStatus.SKIPPED_DONE.value in stdout


def test_main_create_engine_config_error_reports_install_hint(
    work_folder,
    monkeypatch,
    capsys,
) -> None:
    _keep_only(work_folder, {"scanned.pdf"})

    def fake_create_engine(ocr_cfg, log):
        raise ConfigError("rapidocr 引擎依赖缺失，请手动安装：pip install rapidocr>=3.9")

    monkeypatch.setattr("pdf_ocrer.cli.create_engine", fake_create_engine, raising=False)

    exit_code = main([str(work_folder)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "pip install rapidocr>=3.9" in captured.err


def test_main_config_error_writes_default_log(
    work_folder,
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    _keep_only(work_folder, {"scanned.pdf"})
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    config_path = tmp_path / "bad.toml"
    config_path.write_text('[logging]\nlevel = "TRACE"\n', encoding="utf-8")

    exit_code = main(
        [str(work_folder), "--config", str(config_path)],
        engine_factory=lambda ocr_cfg: FakeEngine(),
    )

    _flush_pdf_ocrer_file_handlers()
    captured = capsys.readouterr()
    log_path = tmp_path / "local" / "pdf_ocrer" / "logs" / "pdf_ocrer.log"
    assert exit_code == 1
    assert "设置错误:" in captured.err
    assert "设置错误:" in log_path.read_text(encoding="utf-8")


def test_main_rejects_dpi_override_before_processing(work_folder, capsys) -> None:
    _keep_only(work_folder, {"scanned.pdf"})

    exit_code = main(
        [str(work_folder), "--dpi", "10"],
        engine_factory=lambda ocr_cfg: FakeEngine(),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "设置错误: dpi 超出范围，应为 72–600" in captured.err
    assert not (work_folder / "scanned_OCR.pdf").exists()


def test_pyproject_ships_rapidocr_as_main_dependency() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    deps = pyproject["project"]["dependencies"]
    assert any(dep.startswith("rapidocr>=") for dep in deps)
    assert any(dep.startswith("onnxruntime>=") for dep in deps)
    extras = pyproject["project"]["optional-dependencies"]
    assert "paddle-cpu" not in extras
    assert "rapidocr" not in extras
    assert "rapidocr-gpu-cuda" in extras
    assert "rapidocr-gpu-dml" in extras


def _gt_ocr_lines() -> list[OcrLine]:
    return [OcrLine(text, _px_poly(point, fontsize, text), 0.99) for point, fontsize, text in GT_LINES]


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


def _keep_only(folder: Path, names: set[str]) -> None:
    for path in folder.iterdir():
        if path.is_file() and path.name not in names:
            path.unlink()


def setup_function() -> None:
    _remove_pdf_ocrer_file_handlers()


def teardown_function() -> None:
    _remove_pdf_ocrer_file_handlers()


def _remove_pdf_ocrer_file_handlers() -> None:
    logger = logging.getLogger("pdf_ocrer")
    for handler in list(logger.handlers):
        if getattr(handler, "_pdf_ocrer_file_handler", False):
            logger.removeHandler(handler)
            handler.close()


def _flush_pdf_ocrer_file_handlers() -> None:
    for handler in logging.getLogger("pdf_ocrer").handlers:
        if getattr(handler, "_pdf_ocrer_file_handler", False):
            handler.flush()


def test_configure_utf8_output_reconfigures_streams(monkeypatch):
    from pdf_ocrer.cli import _configure_utf8_output

    calls = []

    class FakeStream:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(sys, "stdout", FakeStream())
    monkeypatch.setattr(sys, "stderr", FakeStream())

    _configure_utf8_output()

    assert calls == [{"encoding": "utf-8"}, {"encoding": "utf-8"}]


def test_configure_utf8_output_tolerates_streams_without_reconfigure(monkeypatch):
    from pdf_ocrer.cli import _configure_utf8_output

    monkeypatch.setattr(sys, "stdout", object())
    monkeypatch.setattr(sys, "stderr", object())

    _configure_utf8_output()  # must not raise
