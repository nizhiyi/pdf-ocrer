"""OCR engine dependency bootstrap.

``create_engine`` calls :func:`ensure_engine_deps` before constructing the
engine, so missing rapidocr/onnxruntime is installed automatically via pip
(source/pip installs only). Frozen (PyInstaller) builds have no usable pip, so
they fail fast with a clear message instead of trying to install.

Detection is module-based (``importlib.util.find_spec``) rather than
distribution-based, matching how the engine itself imports its deps.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable

from pdf_ocrer.config import ConfigError

_ENGINE_REQUIREMENTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # 引擎名 → (需检测的模块, pip 安装参数)。版本与 pyproject.toml 保持一致。
    "rapidocr": (
        ("rapidocr", "onnxruntime"),
        ("rapidocr>=3.9", "onnxruntime>=1.19"),
    ),
}

_INSTALLING_MESSAGE = "正在安装 OCR 引擎依赖…（首次约需几分钟，请保持网络畅通）"
_FROZEN_MESSAGE = (
    "安装版已内置 OCR 引擎但缺少运行组件，请重新安装完整安装包；"
    "或使用源码/pip 方式运行（依赖会自动安装）。"
)


def ensure_engine_deps(
    engine: str,
    log: Callable[[str], None] | None = None,
) -> None:
    """Ensure the named engine's dependencies are importable, installing if needed.

    Raises :class:`ConfigError` when installation fails or is impossible (frozen
    build with a missing engine).
    """
    engine_key = engine.casefold() if isinstance(engine, str) else ""
    requirements = _ENGINE_REQUIREMENTS.get(engine_key)
    if requirements is None:
        return

    modules, packages = requirements
    if _all_modules_available(modules):
        return

    if getattr(sys, "frozen", False):
        raise ConfigError(_FROZEN_MESSAGE)

    install_cmd = " ".join(packages)
    _emit(log, f"未检测到 {engine} 引擎依赖，开始自动安装: pip install {install_cmd}")
    _emit(log, _INSTALLING_MESSAGE)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *packages],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0 or not _all_modules_available(modules):
        _emit(log, f"自动安装失败，请手动执行: pip install {install_cmd}")
        detail = _pip_error_tail(result)
        raise ConfigError(
            f"{engine} 引擎依赖自动安装失败，请手动执行: pip install {install_cmd}"
            + (f"\n{detail}" if detail else "")
        )

    _emit(log, f"{engine} 引擎依赖安装完成")


def _all_modules_available(modules: tuple[str, ...]) -> bool:
    return all(_module_available(name) for name in modules)


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _pip_error_tail(result: subprocess.CompletedProcess[str]) -> str:
    output = result.stderr or result.stdout or ""
    lines = [line for line in output.strip().splitlines() if line.strip()]
    return "\n".join(lines[-3:])


def _emit(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)
