from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pdf_ocrer.config import InputConfig

_logger = logging.getLogger(__name__)

# 本工具生成的输出文件后缀与临时文件标记：扫描时排除，避免把上次的
# 输出（xxx_OCR.pdf）再处理一遍产生 xxx_OCR_OCR.pdf，也把处理中途崩溃
# 残留的 ~*.ocrtmp.pdf 当作输入（下次批次开始时会自动清理这类残留）。
OUTPUT_SUFFIX = "_OCR"
_TEMP_SUFFIX = ".ocrtmp.pdf"


@dataclass(frozen=True)
class ScanItem:
    src: Path
    rel: str


def scan_inputs(folder: Path, cfg: InputConfig) -> list[ScanItem]:
    folder = Path(folder)
    items = _scan_recursive(folder, cfg) if cfg.recursive else _scan_top_level(folder, cfg)
    return sorted(items, key=lambda item: item.rel.casefold())


def allowed_extensions(cfg: InputConfig) -> frozenset[str]:
    return frozenset(
        {"pdf", *(extension.removeprefix(".").casefold() for extension in cfg.image_extensions)}
    )


def is_allowed_input(path: Path, cfg: InputConfig) -> bool:
    return (
        path.is_file()
        and path.suffix.removeprefix(".").casefold() in allowed_extensions(cfg)
        and not is_generated_output(path)
    )


def is_generated_output(path: Path) -> bool:
    name = path.name.casefold()
    if name.startswith("~") and name.endswith(_TEMP_SUFFIX):
        return True
    return path.stem.casefold().endswith(OUTPUT_SUFFIX.casefold())


def _scan_top_level(folder: Path, cfg: InputConfig) -> list[ScanItem]:
    items: list[ScanItem] = []
    try:
        entries = list(folder.iterdir())
    except OSError as exc:
        _logger.warning("无法读取文件夹 path=%s error=%s", folder, exc)
        return items
    for path in entries:
        if not is_allowed_input(path, cfg):
            continue
        items.append(ScanItem(src=path, rel=path.name))
    return items


def _scan_recursive(folder: Path, cfg: InputConfig) -> list[ScanItem]:
    items: list[ScanItem] = []
    visited: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(folder, topdown=True):
        root = Path(dirpath)
        root_resolved = _resolve_path(root)
        if root_resolved in visited:
            dirnames[:] = []
            continue
        visited.add(root_resolved)
        dirnames[:] = _filter_dirnames(root, dirnames, visited)
        for filename in filenames:
            path = root / filename
            if not is_allowed_input(path, cfg):
                continue
            items.append(ScanItem(src=path, rel=path.relative_to(folder).as_posix()))
    return items


def _filter_dirnames(root: Path, dirnames: list[str], visited: set[Path]) -> list[str]:
    kept: list[str] = []
    for name in dirnames:
        child = root / name
        if _resolve_path(child) in visited:
            continue
        kept.append(name)
    return kept


def _resolve_path(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()
