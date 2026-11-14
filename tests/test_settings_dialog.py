from __future__ import annotations

from pathlib import Path

import pytest

tk = pytest.importorskip("tkinter")
ctk = pytest.importorskip("customtkinter")

from pdf_ocrer.config import load_config
from pdf_ocrer.settings_dialog import SettingsDialog


@pytest.fixture()
def root():
    try:
        probe = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    else:
        probe.withdraw()
        probe.destroy()

    instance = ctk.CTk()
    instance.withdraw()
    try:
        yield instance
    finally:
        _destroy_if_exists(instance)


def test_version_switch_updates_model_type_values(root, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)
    dialog = _make_dialog(root, config_path)

    try:
        dialog.ocr_version_var.set("PP-OCRv4")
        dialog._on_version_changed("PP-OCRv4")

        assert set(dialog.model_type_menu.cget("values")) == {"mobile", "server"}
        assert dialog.model_type_var.get() == "mobile"  # v6 small 非法 → 回退 v4 默认

        dialog._on_version_changed("PP-OCRv6")
        assert set(dialog.model_type_menu.cget("values")) == {"tiny", "small", "medium"}
        assert dialog.model_type_var.get() == "small"
    finally:
        _destroy_if_exists(dialog)


def test_invalid_save_shows_error_keeps_window_and_does_not_write(
    root,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, dpi=200)
    original = config_path.read_bytes()
    dialog = _make_dialog(root, config_path)
    errors: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        "pdf_ocrer.settings_dialog.messagebox.showerror",
        lambda title, message, **kwargs: errors.append((title, message, kwargs.get("parent"))),
    )

    try:
        _replace_entry_text(dialog.dpi_entry, "9999")
        dialog._on_save()

        assert errors
        assert errors[0][2] is dialog
        assert _winfo_exists(dialog) == 1
        assert config_path.read_bytes() == original
    finally:
        _destroy_if_exists(dialog)


def test_valid_save_updates_config_and_closes_dialog(root, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)
    dialog = _make_dialog(root, config_path)

    _replace_entry_text(dialog.dpi_entry, "275")
    _replace_entry_text(dialog.min_confidence_entry, "0.42")
    dialog.model_type_var.set("tiny")

    dialog._on_save()

    cfg = load_config(config_path)
    assert _winfo_exists(dialog) == 0
    assert cfg.ocr.ocr_version == "PP-OCRv6"
    assert cfg.ocr.dpi == 275
    assert cfg.ocr.min_confidence == 0.42
    assert cfg.ocr.model_type == "tiny"
    # 废弃的 [performance] 段在保存时自动移除。
    assert "[performance]" not in config_path.read_text(encoding="utf-8")


def test_dialog_saves_device_and_version(root, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)
    dialog = _make_dialog(root, config_path)

    dialog.device_var.set("dml")
    dialog.ocr_version_var.set("PP-OCRv5")
    dialog._on_version_changed("PP-OCRv5")
    dialog.model_type_var.set("server")

    dialog._on_save()

    cfg = load_config(config_path)
    assert cfg.ocr.device == "dml"
    assert cfg.ocr.ocr_version == "PP-OCRv5"
    assert cfg.ocr.model_type == "server"


def test_cancel_closes_dialog_without_writing(root, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, dpi=200)
    original = config_path.read_bytes()
    dialog = _make_dialog(root, config_path)

    _replace_entry_text(dialog.dpi_entry, "300")
    dialog._on_cancel()

    assert _winfo_exists(dialog) == 0
    assert config_path.read_bytes() == original


def _make_dialog(root, config_path: Path) -> SettingsDialog:
    dialog = SettingsDialog(root, config_path)
    dialog.withdraw()
    dialog.update_idletasks()
    return dialog


def _write_config(
    path: Path,
    *,
    dpi: int = 200,
    min_confidence: float = 0.5,
) -> None:
    text = "\n".join(
        [
            "[ocr]",
            f"dpi = {dpi}",
            f"min_confidence = {min_confidence}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _replace_entry_text(entry, text: str) -> None:
    entry.delete(0, "end")
    entry.insert(0, text)


def _winfo_exists(widget) -> int:
    try:
        return int(widget.winfo_exists())
    except tk.TclError:
        return 0


def _destroy_if_exists(widget) -> None:
    if _winfo_exists(widget):
        widget.destroy()
