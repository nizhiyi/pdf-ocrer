from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from pathlib import Path

import pymupdf
from fixtures_gen import GT_LINES, build_image_png

from pdf_ocrer.app_logging import setup_logging
from pdf_ocrer.config import (
    AppConfig,
    DebugConfig,
    InputConfig,
    LoggingConfig,
    OcrConfig,
    OutputConfig,
)
from pdf_ocrer.ocr_engine import OcrLine
from pdf_ocrer.pipeline import FileResult, FileStatus, expand_targets, run_batch
from pdf_ocrer.scanning import is_generated_output

_DPI = 200
_FONT = pymupdf.Font("cjk")


class FakeEngine:
    def __init__(self, lines: list[OcrLine] | None = None) -> None:
        self.lines = _gt_ocr_lines() if lines is None else lines

    def recognize(self, img_rgb) -> list[OcrLine]:
        return self.lines


class FakeDoc:
    def subset_fonts(self) -> None:
        pass

    def save(self, output: Path, garbage: int, deflate: bool) -> None:
        output.write_bytes(b"%PDF-1.7\n")

    def close(self) -> None:
        pass


def make_cfg(**overrides: object) -> AppConfig:
    values = {
        "ocr": OcrConfig(),
        "output": OutputConfig(),
        "debug": DebugConfig(),
        "logging": LoggingConfig(),
    }
    values.update(overrides)
    return AppConfig(**values)


def teardown_function() -> None:
    _remove_pdf_ocrer_file_handlers()


def test_run_batch_statuses_and_preserves_sources(work_folder) -> None:
    _keep_only(work_folder, {"scanned.pdf", "native.pdf", "encrypted.pdf", "corrupt.pdf"})
    cfg = make_cfg()
    before = _source_hashes(work_folder)
    file_results: list[FileResult] = []

    summary = run_batch(
        [work_folder],
        cfg,
        FakeEngine(),
        file_cb=file_results.append,
    )

    assert summary.cancelled is False
    assert [result.source.name for result in summary.results] == [
        "corrupt.pdf",
        "encrypted.pdf",
        "native.pdf",
        "scanned.pdf",
    ]
    by_name = {result.source.name: result for result in summary.results}
    assert by_name["corrupt.pdf"].status is FileStatus.FAILED
    assert by_name["encrypted.pdf"].status is FileStatus.SKIPPED_ENCRYPTED
    assert by_name["native.pdf"].status is FileStatus.SUCCESS_EXISTING_TEXT
    assert by_name["scanned.pdf"].status is FileStatus.SUCCESS_OCR
    assert by_name["native.pdf"].output is None
    assert by_name["scanned.pdf"].output == work_folder / "scanned_OCR.pdf"
    assert not (work_folder / "native_OCR.pdf").exists()
    assert (work_folder / "scanned_OCR.pdf").exists()

    assert _source_hashes(work_folder, exclude_ocr=True) == before
    assert len(file_results) == len(summary.results)
    assert [result.source.name for result in file_results] == [
        "corrupt.pdf",
        "encrypted.pdf",
        "native.pdf",
        "scanned.pdf",
    ]
    assert all(isinstance(result, FileResult) for result in file_results)
    assert [result.status for result in file_results] == [result.status for result in summary.results]


def test_run_batch_output_exists_skips_and_log_message(work_folder) -> None:
    _keep_only(work_folder, {"scanned.pdf"})
    cfg = make_cfg()
    logs: list[str] = []

    first = run_batch([work_folder], cfg, FakeEngine())
    assert [result.status for result in first.results] == [FileStatus.SUCCESS_OCR]

    second = run_batch([work_folder], cfg, FakeEngine(), log_cb=logs.append)

    assert [result.status for result in second.results] == [FileStatus.SKIPPED_DONE]
    assert second.results[0].output == first.results[0].output
    assert second.results[0].rel == "scanned.pdf"
    assert second.results[0].note == ""
    assert logs == ["跳过（已输出）: scanned.pdf"]


def test_run_batch_emits_per_file_result_logs(work_folder) -> None:
    _keep_only(work_folder, {"scanned.pdf", "native.pdf"})
    logs: list[str] = []

    run_batch([work_folder], make_cfg(), FakeEngine(), log_cb=logs.append)

    assert any(line.startswith("OCR 完成: scanned.pdf") for line in logs)
    assert any(line.startswith("跳过（已OCR）: native.pdf") for line in logs)


def test_run_batch_force_overwrites_existing_output(work_folder) -> None:
    _keep_only(work_folder, {"scanned.pdf"})
    cfg = make_cfg()
    pre_existing = work_folder / "scanned_OCR.pdf"
    pre_existing.write_bytes(b"%PDF-1.7 previous output\n")

    summary = run_batch([work_folder], cfg, FakeEngine(), force=True)

    assert [result.status for result in summary.results] == [FileStatus.SUCCESS_OCR]
    assert summary.results[0].output == work_folder / "scanned_OCR.pdf"
    # 旧输出被覆盖，而不是生成带序号的新文件。
    assert pre_existing.read_bytes() != b"%PDF-1.7 previous output\n"
    assert not (work_folder / "scanned_OCR_2.pdf").exists()


def test_run_batch_warns_when_medium_model_selected(work_folder) -> None:
    _keep_only(work_folder, {"scanned.pdf"})
    logs: list[str] = []
    cfg = make_cfg(ocr=OcrConfig(ocr_version="PP-OCRv6", model_type="medium"))

    run_batch([work_folder], cfg, FakeEngine(), log_cb=logs.append)

    assert any("PP-OCRv6 medium" in line for line in logs)


def test_run_batch_pending_cb_lists_all_files_before_processing(work_folder) -> None:
    _keep_only(work_folder, {"scanned.pdf", "native.pdf"})
    pending: list[list[str]] = []

    summary = run_batch(
        [work_folder],
        make_cfg(),
        FakeEngine(),
        pending_cb=pending.append,
    )

    assert pending == [["native.pdf", "scanned.pdf"]]
    assert len(summary.results) == 2


def test_run_batch_empty_targets_returns_empty_summary(tmp_path) -> None:
    logs: list[str] = []

    summary = run_batch([tmp_path], make_cfg(), FakeEngine(), log_cb=logs.append)

    assert summary.results == []
    assert summary.cancelled is False
    assert logs == ["找不到可处理的 PDF/图片文件。"]


def test_run_batch_writes_batch_and_file_status_lines_to_log(work_folder, tmp_path) -> None:
    _keep_only(work_folder, {"scanned.pdf", "native.pdf"})
    log_dir = tmp_path / "logs"
    cfg = make_cfg(logging=LoggingConfig(dir=str(log_dir)))
    setup_logging(cfg.logging)

    summary = run_batch([work_folder], cfg, FakeEngine())
    second = run_batch([work_folder], cfg, FakeEngine())
    _flush_pdf_ocrer_file_handlers()

    text = (log_dir / "pdf_ocrer.log").read_text(encoding="utf-8")
    assert "batch start" in text
    for result in summary.results:
        output_name = "" if result.output is None else result.output.name
        assert f"source={result.source.name}" in text
        assert f"status={result.status.value}" in text
        assert f"output={output_name}" in text
    assert "batch end" in text
    assert "results=2" in text
    assert [result.status for result in second.results] == [
        FileStatus.SUCCESS_EXISTING_TEXT,
        FileStatus.SKIPPED_DONE,
    ]


def test_run_batch_recursive_outputs_next_to_sources(work_folder, fixtures_dir) -> None:
    _keep_only(work_folder, set())
    (work_folder / "alpha").mkdir()
    (work_folder / "beta").mkdir()
    shutil.copy2(fixtures_dir / "native.pdf", work_folder / "alpha" / "a.pdf")
    shutil.copy2(fixtures_dir / "native.pdf", work_folder / "beta" / "a.pdf")
    nested_dir = work_folder / "alpha" / "nested"
    nested_dir.mkdir(parents=True)
    shutil.copy2(fixtures_dir / "native.pdf", nested_dir / "b.pdf")
    progress_names: list[str] = []
    cfg = make_cfg(input=InputConfig(recursive=True))

    summary = run_batch(
        [work_folder],
        cfg,
        FakeEngine(),
        progress_cb=lambda file_i, file_n, page_i, page_n, name: progress_names.append(name),
    )

    assert summary.cancelled is False
    assert [result.rel for result in summary.results] == [
        "alpha/a.pdf",
        "alpha/nested/b.pdf",
        "beta/a.pdf",
    ]
    assert all(result.status is FileStatus.SUCCESS_EXISTING_TEXT for result in summary.results)
    assert all(result.output is None for result in summary.results)
    assert not (work_folder / "alpha" / "a_OCR.pdf").exists()
    assert not (work_folder / "alpha" / "nested" / "b_OCR.pdf").exists()
    assert not (work_folder / "beta" / "a_OCR.pdf").exists()
    assert progress_names == ["alpha/a.pdf", "alpha/nested/b.pdf", "beta/a.pdf"]


def test_expand_targets_mixed_files_and_folders_disambiguates_rel(tmp_path: Path, fixtures_dir) -> None:
    folder_a = tmp_path / "A"
    folder_b = tmp_path / "B"
    folder_a.mkdir()
    folder_b.mkdir()
    shutil.copy2(fixtures_dir / "native.pdf", folder_a / "a.pdf")
    shutil.copy2(fixtures_dir / "native.pdf", folder_b / "a.pdf")
    shutil.copy2(fixtures_dir / "native.pdf", tmp_path / "solo.pdf")
    (tmp_path / "note.txt").write_bytes(b"txt")

    items = expand_targets([folder_a, tmp_path / "solo.pdf", folder_b], InputConfig())

    assert [(item.rel, item.src.name) for item in items] == [
        ("A/a.pdf", "a.pdf"),
        ("B/a.pdf", "a.pdf"),
        ("solo.pdf", "solo.pdf"),
    ]


def test_expand_targets_ignores_unsupported_explicit_files(tmp_path: Path) -> None:
    (tmp_path / "doc.docx").write_bytes(b"docx")
    (tmp_path / "real.pdf").write_bytes(b"%PDF-1.7\n")

    items = expand_targets([tmp_path / "doc.docx", tmp_path / "real.pdf"], InputConfig())

    assert [item.rel for item in items] == ["real.pdf"]


def test_run_batch_missing_explicit_file_fails_and_continues(work_folder: Path) -> None:
    _keep_only(work_folder, {"native.pdf"})
    missing = work_folder / "missing.pdf"
    targets = [missing, work_folder / "native.pdf"]
    logs: list[str] = []

    summary = run_batch(targets, make_cfg(), FakeEngine(), log_cb=logs.append)

    assert summary.cancelled is False
    assert [result.rel for result in summary.results] == ["missing.pdf", "native.pdf"]
    assert [result.status for result in summary.results] == [
        FileStatus.FAILED,
        FileStatus.SUCCESS_EXISTING_TEXT,
    ]
    assert "文件不存在" in summary.results[0].note
    assert summary.results[1].output is None
    # 失败文件必须在运行日志中给出原因，而不是静默消失。
    assert any(line.startswith("失败: missing.pdf") for line in logs)


def test_run_batch_image_input_writes_searchable_pdf(work_folder: Path) -> None:
    _keep_only(work_folder, set())
    build_image_png(work_folder / "scan.png")
    cfg = make_cfg()

    summary = run_batch([work_folder], cfg, FakeEngine())

    assert summary.cancelled is False
    assert len(summary.results) == 1
    result = summary.results[0]
    assert result.rel == "scan.png"
    assert result.status is FileStatus.SUCCESS_OCR
    assert result.output == work_folder / "scan_OCR.pdf"
    assert result.output.exists()
    doc = pymupdf.open(result.output)
    assert "诊断证明书" in doc[0].get_text()
    doc.close()


def test_run_batch_encrypted_retried_every_run_without_manifest(work_folder: Path) -> None:
    _keep_only(work_folder, {"encrypted.pdf"})
    cfg = make_cfg()

    first = run_batch([work_folder], cfg, FakeEngine())
    second = run_batch([work_folder], cfg, FakeEngine())

    assert [result.status for result in first.results] == [FileStatus.SKIPPED_ENCRYPTED]
    assert [result.status for result in second.results] == [FileStatus.SKIPPED_ENCRYPTED]
    assert second.results[0].output is None


def test_run_batch_resume_after_cancel_skips_completed(work_folder: Path) -> None:
    _keep_only(work_folder, {"native.pdf", "scanned.pdf"})
    cfg = make_cfg()
    cancel = threading.Event()

    def cancel_after_first(file_i: int, file_n: int, page_i: int, page_n: int, filename: str) -> None:
        if file_i == 1 and page_i == page_n:
            cancel.set()

    first = run_batch(
        [work_folder],
        cfg,
        FakeEngine(),
        progress_cb=cancel_after_first,
        cancel_event=cancel,
    )
    second = run_batch([work_folder], cfg, FakeEngine())

    assert first.cancelled is True
    assert [result.source.name for result in first.results] == ["native.pdf"]
    assert [result.status for result in second.results] == [
        FileStatus.SUCCESS_EXISTING_TEXT,
        FileStatus.SUCCESS_OCR,
    ]


class _SaveFailsDoc:
    """save 中途抛异常的假文档：模拟输出写到一半失败。"""

    def subset_fonts(self) -> None:
        pass

    def save(self, output: Path, garbage: int, deflate: bool) -> None:
        output.write_bytes(b"%PDF-1.7\n% partial")
        raise RuntimeError("disk full while saving")

    def close(self) -> None:
        pass


def test_run_batch_save_failure_leaves_no_partial_output(work_folder, monkeypatch) -> None:
    _keep_only(work_folder, {"scanned.pdf"})
    output = work_folder / "scanned_OCR.pdf"

    class _PartialProcessed:
        def __init__(self) -> None:
            self.reports = [type("R", (), {"action": "ocr"})()]
            self.total_pages = 1
            self.ocr_pages = 0
            self.doc = _SaveFailsDoc()

    def fake_process_pdf(*args, **kwargs):
        return _PartialProcessed()

    monkeypatch.setattr("pdf_ocrer.pipeline.process_pdf", fake_process_pdf)
    logs: list[str] = []

    summary = run_batch([work_folder], make_cfg(), FakeEngine(), log_cb=logs.append)

    assert [result.status for result in summary.results] == [FileStatus.FAILED]
    # 半截内容只落在临时文件里且必须被清理：最终输出不存在、也无临时残留。
    assert not output.exists()
    assert list(work_folder.glob("~*.ocrtmp.pdf")) == []
    assert any("disk full while saving" in line for line in logs)


def test_expand_targets_rejects_generated_output(tmp_path) -> None:
    """显式传入的 *_OCR.pdf 是本工具的输出文件：一律排除，绝不作为输入。"""
    pdf = tmp_path / "报告.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    ocr_output = tmp_path / "报告_OCR.pdf"
    ocr_output.write_bytes(b"%PDF-1.7\n")

    items = expand_targets([pdf, ocr_output], InputConfig())

    assert [item.src for item in items] == [pdf]


def test_overwrite_rejects_generated_output_source(work_folder) -> None:
    """覆盖模式防线：源文件本身是 *_OCR.pdf 时拒绝覆盖并保留输出。"""
    from pdf_ocrer.pipeline import _overwrite_after_finalize

    src = work_folder / "报告_OCR.pdf"
    src.write_bytes(b"%PDF-1.7\n% previous output")
    output = work_folder / "报告_OCR_OCR.pdf"
    output.write_bytes(b"%PDF-1.7\n% new ocr result")

    note = _overwrite_after_finalize(src, output)

    assert note is not None
    assert "输出文件" in note
    # 源文件原样保留，新输出也不被消耗。
    assert src.read_bytes() == b"%PDF-1.7\n% previous output"
    assert output.exists()


def test_run_batch_no_text_produces_no_output_and_skips_overwrite(
    work_folder,
    monkeypatch,
) -> None:
    """OCR 未识别出任何文字：不产出 _OCR、覆盖模式也不备份/替换原文件。"""
    _keep_only(work_folder, {"scanned.pdf"})

    class _EmptyReport:
        action = "empty"
        line_count = 0

    class _NoTextDoc(FakeDoc):
        pass

    class _NoTextProcessed:
        def __init__(self) -> None:
            self.reports = [_EmptyReport()]
            self.total_pages = 1
            self.ocr_pages = 0
            self.doc = _NoTextDoc()

    monkeypatch.setattr(
        "pdf_ocrer.pipeline.process_pdf",
        lambda *args, **kwargs: _NoTextProcessed(),
    )
    backups = _install_fake_backup(monkeypatch, work_folder)
    logs: list[str] = []
    cfg = make_cfg(output=OutputConfig(overwrite_original=True))

    summary = run_batch(
        [work_folder],
        cfg,
        FakeEngine(),
        log_cb=logs.append,
    )

    result = summary.results[0]
    assert result.status is FileStatus.NO_TEXT_FOUND
    assert result.output is None
    assert not (work_folder / "scanned_OCR.pdf").exists()
    # 无输出 → 不备份、不覆盖，原文件原样保留。
    assert backups == []
    assert any(line.startswith("跳过（无文本）") for line in logs)


def _install_fake_backup(monkeypatch, target_root: Path) -> list[Path]:
    """把备份重定向到测试目录，返回已发生的备份路径列表。"""
    import shutil as _shutil

    from pdf_ocrer import backup as backup_mod

    backups: list[Path] = []

    def fake_backup_original(src: Path) -> Path:
        backup_dir = target_root / "downloads" / "pdf-ocrer"
        backup_dir.mkdir(parents=True, exist_ok=True)
        dest = backup_dir / src.name
        _shutil.copy2(src, dest)
        backups.append(dest)
        return dest

    monkeypatch.setattr(backup_mod, "backup_original", fake_backup_original)
    return backups


def test_run_batch_overwrite_original_replaces_source_and_backs_up(
    work_folder, tmp_path, monkeypatch
) -> None:
    _keep_only(work_folder, {"scanned.pdf"})
    before = _source_hashes(work_folder)
    backups = _install_fake_backup(monkeypatch, tmp_path)
    cfg = make_cfg(output=OutputConfig(overwrite_original=True))

    summary = run_batch([work_folder], cfg, FakeEngine())

    result = summary.results[0]
    assert result.status is FileStatus.SUCCESS_OCR
    assert result.output == work_folder / "scanned.pdf"
    assert result.note == ""
    # 原文件已被 OCR 版替换（内容改变）。
    assert _source_hashes(work_folder) != before
    doc = pymupdf.open(work_folder / "scanned.pdf")
    assert "诊断证明书" in doc[0].get_text()
    doc.close()
    # 原文件已备份（内容与原文件一致），_OCR 文件不存在。
    assert len(backups) == 1
    assert hashlib.sha256(backups[0].read_bytes()).hexdigest() == before["scanned.pdf"]
    assert not (work_folder / "scanned_OCR.pdf").exists()


def test_run_batch_overwrite_backup_failure_keeps_ocr_output(
    work_folder, tmp_path, monkeypatch
) -> None:
    from pdf_ocrer import backup as backup_mod

    _keep_only(work_folder, {"scanned.pdf"})
    before = _source_hashes(work_folder)

    def failing_backup(src: Path) -> Path:
        raise OSError("备份文件夹不可写")

    monkeypatch.setattr(backup_mod, "backup_original", failing_backup)
    cfg = make_cfg(output=OutputConfig(overwrite_original=True))

    summary = run_batch([work_folder], cfg, FakeEngine())

    result = summary.results[0]
    assert result.status is FileStatus.SUCCESS_OCR
    assert result.output == work_folder / "scanned_OCR.pdf"
    assert "覆盖原文件失败" in result.note
    # 原文件未动（排除新生成的 _OCR 后哈希一致），_OCR 文件保留。
    assert _source_hashes(work_folder, exclude_ocr=True) == before
    assert (work_folder / "scanned_OCR.pdf").exists()


def test_run_batch_overwrite_replace_failure_keeps_ocr_output(
    work_folder, tmp_path, monkeypatch
) -> None:
    _keep_only(work_folder, {"scanned.pdf"})
    before = _source_hashes(work_folder)
    _install_fake_backup(monkeypatch, tmp_path)

    real_replace = os.replace

    def failing_replace(src, dst):
        if Path(dst).name == "scanned.pdf":
            raise PermissionError(32, "文件被占用")
        return real_replace(src, dst)

    monkeypatch.setattr("pdf_ocrer.pipeline.os.replace", failing_replace)
    cfg = make_cfg(output=OutputConfig(overwrite_original=True))

    summary = run_batch([work_folder], cfg, FakeEngine())

    result = summary.results[0]
    assert result.status is FileStatus.SUCCESS_OCR
    assert result.output == work_folder / "scanned_OCR.pdf"
    assert "覆盖原文件失败" in result.note
    # 原文件未动（排除新生成的 _OCR 后哈希一致），_OCR 文件保留。
    assert _source_hashes(work_folder, exclude_ocr=True) == before
    assert (work_folder / "scanned_OCR.pdf").exists()


def test_run_batch_overwrite_mode_does_not_skip_existing_ocr_output(work_folder, tmp_path, monkeypatch) -> None:
    _keep_only(work_folder, {"scanned.pdf"})
    _install_fake_backup(monkeypatch, tmp_path)
    run_batch([work_folder], make_cfg(), FakeEngine())
    assert (work_folder / "scanned_OCR.pdf").exists()

    cfg = make_cfg(output=OutputConfig(overwrite_original=True))
    summary = run_batch([work_folder], cfg, FakeEngine())

    # 覆盖模式不做 _OCR 存在性跳过：原文件被替换。
    result = summary.results[0]
    assert result.status is FileStatus.SUCCESS_OCR
    assert result.output == work_folder / "scanned.pdf"
    assert not (work_folder / "scanned_OCR.pdf").exists()


def test_run_batch_overwrite_mode_still_skips_existing_image_output(
    work_folder, tmp_path, monkeypatch
) -> None:
    # 图片不能覆盖原文件：_OCR 已存在时不应重复 OCR。
    _keep_only(work_folder, set())
    build_image_png(work_folder / "scan.png")
    _install_fake_backup(monkeypatch, tmp_path)
    run_batch([work_folder], make_cfg(), FakeEngine())
    assert (work_folder / "scan_OCR.pdf").exists()

    ocr_before = (work_folder / "scan_OCR.pdf").read_bytes()
    cfg = make_cfg(output=OutputConfig(overwrite_original=True))
    summary = run_batch([work_folder], cfg, FakeEngine())

    result = summary.results[0]
    assert result.status is FileStatus.SKIPPED_DONE
    assert (work_folder / "scan_OCR.pdf").read_bytes() == ocr_before


def test_run_batch_existing_text_file_is_never_overwritten(work_folder, tmp_path, monkeypatch) -> None:
    from pdf_ocrer import backup as backup_mod

    _keep_only(work_folder, {"native.pdf"})
    before = _source_hashes(work_folder)
    called: list[Path] = []

    def spy_backup(src: Path) -> Path:
        called.append(src)
        raise AssertionError("已有文字层文件不应触发备份")

    monkeypatch.setattr(backup_mod, "backup_original", spy_backup)
    cfg = make_cfg(output=OutputConfig(overwrite_original=True))

    summary = run_batch([work_folder], cfg, FakeEngine())

    result = summary.results[0]
    assert result.status is FileStatus.SUCCESS_EXISTING_TEXT
    assert result.output is None
    assert _source_hashes(work_folder) == before
    assert called == []


def test_run_batch_image_input_never_overwrites_original(work_folder, tmp_path, monkeypatch) -> None:
    _keep_only(work_folder, set())
    build_image_png(work_folder / "scan.png")
    before = (work_folder / "scan.png").read_bytes()
    called: list[Path] = []

    from pdf_ocrer import backup as backup_mod

    def spy_backup(src: Path) -> Path:
        called.append(src)
        raise AssertionError("图片输入不应触发备份/覆盖")

    monkeypatch.setattr(backup_mod, "backup_original", spy_backup)
    cfg = make_cfg(output=OutputConfig(overwrite_original=True))

    summary = run_batch([work_folder], cfg, FakeEngine())

    result = summary.results[0]
    # 图片输入退化为导出 _OCR：扩展名不能变成 .png 内容却是 PDF。
    assert result.status is FileStatus.SUCCESS_OCR
    assert result.output == work_folder / "scan_OCR.pdf"
    assert "图片输入不支持覆盖原文件" in result.note
    assert (work_folder / "scan_OCR.pdf").exists()
    assert (work_folder / "scan.png").read_bytes() == before
    assert called == []


def test_is_generated_output_matches_suffix_and_temp_files(tmp_path: Path) -> None:

    assert is_generated_output(tmp_path / "doc_OCR.pdf") is True
    assert is_generated_output(tmp_path / "doc_ocr.PDF") is True
    assert is_generated_output(tmp_path / "~abc123.ocrtmp.pdf") is True
    assert is_generated_output(tmp_path / "doc.pdf") is False
    assert is_generated_output(tmp_path / "OCR_notes.pdf") is False


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


def _source_hashes(folder: Path, exclude_ocr: bool = False) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.glob("*.pdf"))
        if not (exclude_ocr and is_generated_output(path))
    }


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
