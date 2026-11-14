"""Guardrails for the Windows packaging artifacts (packaging/).

These run in CI's release gate so a broken installer config, a stale pyproject
version wiring, or a paddle leak into the packaged config is caught before a tag
is ever cut — without needing to actually build the installer.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pdf_ocrer.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGING = REPO_ROOT / "packaging"


def test_installer_config_is_valid_and_defaults_match_bundle() -> None:
    cfg = load_config(PACKAGING / "config.installer.toml")

    # 安装包只内置 PP-OCRv6 small，默认配置必须与之匹配（离线可用）。
    assert cfg.ocr.ocr_version == "PP-OCRv6"
    assert cfg.ocr.model_type == "small"
    assert cfg.ocr.device == "cpu"
    # The installer seed ships only the OCR section: no naming/llm/watch remnants.
    assert not hasattr(cfg, "naming")
    assert not hasattr(cfg, "llm")
    assert not hasattr(cfg, "watch")


def test_pyproject_version_is_dynamic_from_init() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "version" in pyproject["project"]["dynamic"]
    assert "version" not in pyproject["project"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/pdf_ocrer/__init__.py"


def test_packaging_files_exist() -> None:
    for name in ("pdf_ocrer.spec", "installer.iss", "build.ps1", "entry_gui.py"):
        assert (PACKAGING / name).is_file(), f"missing packaging/{name}"
    # 本地一键打包脚本（双击运行）
    assert (REPO_ROOT / "打包安装包.bat").is_file(), "missing one-click build script"


def test_installer_runs_without_administrator_privileges() -> None:
    text = (PACKAGING / "installer.iss").read_text(encoding="utf-8")

    assert "PrivilegesRequired=lowest" in text
    assert "PrivilegesRequired=admin" not in text
    # 默认安装目录必须是当前用户可写位置，绝不能是 Program Files。
    assert "DefaultDirName={localappdata}" in text
    assert "DefaultDirName={autopf}" not in text


def test_installer_spec_excludes_paddle() -> None:
    spec = (PACKAGING / "pdf_ocrer.spec").read_text(encoding="utf-8")

    for pkg in ("paddleocr", "paddlepaddle", "paddlex"):
        assert pkg in spec, f"spec should list {pkg} in excludes"
    assert 'collect_data_files("rapidocr")' in spec


def test_app_icon_wired_into_packaging() -> None:
    """应用图标必须存在并接入 PyInstaller/Inno/运行时三处。"""
    icon = PACKAGING / "app.ico"
    assert icon.is_file(), "missing packaging/app.ico"

    spec = (PACKAGING / "pdf_ocrer.spec").read_text(encoding="utf-8")
    assert "app.ico" in spec, "spec must reference packaging/app.ico"

    iss = (PACKAGING / "installer.iss").read_text(encoding="utf-8")
    assert "SetupIconFile=app.ico" in iss, "installer must use packaging/app.ico"


def test_release_workflow_present() -> None:
    workflow = REPO_ROOT / ".github" / "workflows" / "release.yml"
    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert "build.ps1" in text
    assert "__version__" in text  # tag/version consistency gate


def test_frozen_entry_points_call_freeze_support() -> None:
    # PyInstaller + multiprocessing spawn requires freeze_support() at the very
    # start of every frozen entry point, or the OCR worker process recursively
    # relaunches the app.
    for rel in ("packaging/entry_gui.py", "src/pdf_ocrer/__main__.py"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "freeze_support()" in text, f"{rel} must call multiprocessing.freeze_support()"
