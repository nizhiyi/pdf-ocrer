"""Tests for the original-file backup helpers (backup.py).

The real SHGetKnownFolderPath call is verified manually on a redirected
profile; these tests monkeypatch ``_known_folder_downloads`` to exercise the
resolution order, fallback, and backup actions without touching the system.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

from pdf_ocrer import backup as backup_mod


@pytest.fixture()
def fake_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    monkeypatch.setattr(backup_mod, "_known_folder_downloads", lambda: downloads)
    return downloads


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only resolution path")
def test_downloads_guid_constant_is_the_verified_variant() -> None:
    # 项目验证过的下载 GUID（重定向 profile 环境实测有效）。
    assert backup_mod._DOWNLOADS_GUID == "{374DE290-123F-4565-9164-39C4925E467B}"
    uuid.UUID(backup_mod._DOWNLOADS_GUID)  # 必须是合法 GUID


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only resolution path")
def test_downloads_dir_uses_known_folder_when_available(fake_downloads: Path) -> None:
    assert backup_mod.downloads_dir() == fake_downloads


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only resolution path")
def test_downloads_dir_falls_back_to_home_when_known_folder_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backup_mod, "_known_folder_downloads", lambda: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path, raising=False)

    assert backup_mod.downloads_dir() == tmp_path / "Downloads"


def test_backup_root_is_pdf_ocrer_subdir(fake_downloads: Path) -> None:
    assert backup_mod.backup_root() == fake_downloads / "pdf-ocrer备份"


def test_unique_backup_path_returns_plain_name_when_free(tmp_path: Path) -> None:
    assert backup_mod.unique_backup_path(tmp_path, "报告.pdf") == tmp_path / "报告.pdf"


def test_unique_backup_path_adds_timestamp_on_conflict(tmp_path: Path) -> None:
    (tmp_path / "报告.pdf").touch()

    result = backup_mod.unique_backup_path(tmp_path, "报告.pdf")

    assert result.parent == tmp_path
    assert result.name.startswith("报告_")
    assert result.suffix == ".pdf"
    assert result != tmp_path / "报告.pdf"


def test_backup_original_copies_content(tmp_path: Path, fake_downloads: Path) -> None:
    src = tmp_path / "scan.pdf"
    src.write_bytes(b"%PDF-1.7 original bytes")

    backup = backup_mod.backup_original(src)

    assert backup == fake_downloads / "pdf-ocrer备份" / "scan.pdf"
    assert backup.read_bytes() == b"%PDF-1.7 original bytes"
    # 原文件保留在原处。
    assert src.exists()
