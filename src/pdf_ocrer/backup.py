"""Original-file backup for the "overwrite original" mode.

When the user enables 覆盖原文件, the OCR result replaces the source file. The
source is backed up first so nothing is ever lost. Backups live under the
user's real Downloads folder (resolved via SHGetKnownFolderPath, which follows
Explorer's redirection setting), in a ``pdf-ocrer`` subfolder.

Resolution order for the Downloads folder:
1. ``SHGetKnownFolderPath`` with the Downloads GUID
   ``{374DE290-123F-4565-9164-39C4925E467B}`` (observed in redirected-profile
   environments; verified to resolve the real Downloads folder there);
2. ``~/Downloads``.
"""

from __future__ import annotations

import ctypes
import logging
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

_logger = logging.getLogger(__name__)

_BACKUP_SUBDIR = "pdf-ocrer备份"
# 本项目验证可用的下载文件夹 GUID（重定向 profile 环境实测有效）。
_DOWNLOADS_GUID = "{374DE290-123F-4565-9164-39C4925E467B}"


def downloads_dir() -> Path:
    """The user's Downloads folder, following system redirection."""
    if sys.platform == "win32":
        known = _known_folder_downloads()
        if known is not None:
            return known

    return Path.home() / "Downloads"


def _known_folder_downloads() -> Path | None:
    class _GUID(ctypes.Structure):
        _fields_ = [
            ("d1", ctypes.c_ulong),
            ("d2", ctypes.c_ushort),
            ("d3", ctypes.c_ushort),
            ("d4", ctypes.c_ubyte * 8),
        ]

    try:
        fid = _GUID.from_buffer_copy(uuid.UUID(_DOWNLOADS_GUID).bytes_le)
        buf = ctypes.c_wchar_p()
        result = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(fid), 0, None, ctypes.byref(buf)
        )
        if result == 0 and buf.value:
            return Path(buf.value)
        # 部分环境未注册该 known folder，降级 debug 后走回退路径。
        _logger.debug("SHGetKnownFolderPath failed hr=0x%08X", result & 0xFFFFFFFF)
    except (OSError, ValueError, AttributeError) as exc:
        _logger.warning("known-folder lookup failed: %s", exc)
    return None


def backup_root() -> Path:
    """Backup folder: <Downloads>/pdf-ocrer."""
    return downloads_dir() / _BACKUP_SUBDIR


def unique_backup_path(backup_dir: Path, filename: str) -> Path:
    """Unique target path; conflicts get a timestamp suffix (报告_20261112_135012.pdf)."""
    candidate = backup_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return backup_dir / f"{stem}_{stamp}{suffix}"


def backup_original(src: Path) -> Path:
    """Copy the source file into the backup folder and return the backup path."""
    backup_dir = backup_root()
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = unique_backup_path(backup_dir, src.name)
    shutil.copy2(src, backup)
    _logger.info("original backed up src=%s backup=%s", src, backup)
    return backup
