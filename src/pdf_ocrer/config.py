from __future__ import annotations

import os
import shutil
import sys
import tomllib
import warnings
from collections.abc import MutableMapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

import tomlkit
from tomlkit.exceptions import TOMLKitError


@dataclass(frozen=True)
class OcrConfig:
    dpi: int = 200
    min_confidence: float = 0.5
    skip_pages_with_text: bool = True
    min_existing_chars: int = 30
    device: str = "cpu"
    det_limit_side_len: int | None = None
    # OCR 模型版本与大小。PP-OCRv6 只有 tiny/small/medium；v4/v5 只有
    # mobile/server（与 RapidOCR default_models.yaml 一致）。默认 v6 small。
    ocr_version: str = "PP-OCRv6"
    model_type: str = "small"
    cpu_threads: int = 0
    textline_orientation: bool = True


@dataclass(frozen=True)
class OutputConfig:
    # 覆盖原文件模式：OCR 结果直接替换原文件，原文件备份到下载文件夹。
    overwrite_original: bool = False


@dataclass(frozen=True)
class InputConfig:
    recursive: bool = False
    image_extensions: tuple[str, ...] = ("jpg", "jpeg", "png", "tif", "tiff")


@dataclass(frozen=True)
class DebugConfig:
    visible_text: bool = False


@dataclass(frozen=True)
class GuiConfig:
    appearance: str = "system"


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool = True
    level: str = "INFO"
    dir: str = ""


@dataclass(frozen=True)
class AppConfig:
    ocr: OcrConfig
    output: OutputConfig
    debug: DebugConfig
    input: InputConfig = InputConfig()
    gui: GuiConfig = GuiConfig()
    logging: LoggingConfig = LoggingConfig()


class ConfigError(ValueError):
    """Raised when TOML config values cannot be used by the application."""


def _write_config_text(config_path: Path, content: str) -> None:
    """原子写配置：临时文件 + os.replace，避免读到截断中的半截 TOML。"""
    temp = config_path.parent / f"~{uuid4().hex}.ocrtmp.toml"
    try:
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, config_path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class CommonSettings:
    ocr_version: str = "PP-OCRv6"
    dpi: int = 200
    min_confidence: float = 0.5
    device: str = "cpu"
    model_type: str = "small"


_T = TypeVar("_T")

# 旧版本配置中已废弃的段：加载时静默忽略，不告警不报错。
# performance：0.8 及以前的「同时处理文件数」，0.9 起固定单文件顺序处理。
_DEPRECATED_SECTIONS = frozenset({"naming", "llm", "watch", "performance"})

# 旧版本 [ocr] 段中已废弃的键（paddle 引擎时代遗留）：静默忽略。
_DEPRECATED_OCR_KEYS = frozenset(
    {"engine", "lang", "enable_mkldnn", "det_model_name", "rec_model_name"}
)


def load_config(path: Path | None = None) -> AppConfig:
    config_path = Path("config.toml") if path is None else path
    if not config_path.exists():
        if getattr(sys, "frozen", False):
            return _validate(_frozen_default_config())
        return _validate(AppConfig(OcrConfig(), OutputConfig(), DebugConfig()))

    try:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"设置文件 TOML 格式错误: {exc}") from exc

    sections: dict[str, type[object]] = {
        "ocr": OcrConfig,
        "output": OutputConfig,
        "input": InputConfig,
        "debug": DebugConfig,
        "gui": GuiConfig,
        "logging": LoggingConfig,
    }
    section_values: dict[str, object] = {}

    for section_name, section_data in raw.items():
        if section_name in _DEPRECATED_SECTIONS:
            continue
        section_cls = sections.get(section_name)
        if section_cls is None:
            _warn_unknown(section_name)
            continue
        if not isinstance(section_data, dict):
            raise ConfigError(f"设置项 {section_name} 必须是 TOML 表格")
        section_values[section_name] = _build_section(section_cls, section_name, section_data)

    return _validate(
        AppConfig(
            ocr=_section(section_values, "ocr", OcrConfig()),
            output=_section(section_values, "output", OutputConfig()),
            debug=_section(section_values, "debug", DebugConfig()),
            input=_section(section_values, "input", InputConfig()),
            gui=_section(section_values, "gui", GuiConfig()),
            logging=_section(section_values, "logging", LoggingConfig()),
        )
    )


def ensure_config_file(path: Path) -> None:
    config_path = Path(path)
    if config_path.exists():
        return

    if frozen_data_dir() is not None:
        # Frozen build: seed from the bundle, and if that is unavailable fall back
        # to the minimal seed — never an empty file, which would make load_config
        # skip the frozen fallback and silently use in-memory defaults.
        bootstrap_frozen_config(config_path)
        if config_path.exists():
            return
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(_FROZEN_FALLBACK_CONFIG, encoding="utf-8")
        except OSError:
            pass
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)
    example_path = Path("config.example.toml")
    if example_path.exists():
        shutil.copyfile(example_path, config_path)
        return

    config_path.write_text("", encoding="utf-8")


def frozen_data_dir() -> Path | None:
    """User-writable app-data dir when running as a frozen (packaged) app, else None.

    A packaged install lives under Program Files, which is not user-writable and
    whose location is independent of the process working directory. Keeping the
    config there would break settings saves for non-admin users and would make a
    launch from any folder other than the install dir miss the config entirely.
    So the frozen app keeps its config under %APPDATA%\\pdf_ocrer instead.
    """
    if not getattr(sys, "frozen", False):
        return None
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "pdf_ocrer"
    return Path.home() / ".pdf_ocrer"


def default_config_path() -> Path:
    """Where the CLI/GUI look for config.toml when no explicit path is given.

    Frozen app: per-user %APPDATA%\\pdf_ocrer\\config.toml. Source/dev: config.toml
    relative to the working directory (unchanged behavior).
    """
    data_dir = frozen_data_dir()
    return (data_dir / "config.toml") if data_dir is not None else Path("config.toml")


def _bundled_resource(name: str) -> Path | None:
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) / name if base else None


# Minimal fail-safe seed for the frozen app. The packaged build ships only the
# RapidOCR engine (PP-OCRv6 small), so if the bundled template can't be copied
# we must still land on valid defaults.
_FROZEN_FALLBACK_CONFIG = "[ocr]\n"


def _frozen_default_config() -> AppConfig:
    """In-memory default a frozen build uses when no config file is present.

    The frozen build ships only RapidOCR, whose defaults (PP-OCRv6 small) match
    :class:`OcrConfig` defaults. This is the last-resort safety net if even
    writing the seed failed.
    """
    return AppConfig(
        OcrConfig(),
        OutputConfig(),
        DebugConfig(),
    )


def bootstrap_frozen_config(path: Path) -> None:
    """On a frozen app's first run, seed the per-user config.

    No-op when not frozen or when the config already exists (never overwrites a
    user's customized config). When frozen and the config is missing it is always
    created: from the bundled template if available, otherwise from a minimal
    fallback with in-memory defaults.
    """
    if frozen_data_dir() is None:
        return
    config_path = Path(path)
    if config_path.exists():
        return

    source = _bundled_resource("config.installer.toml") or _bundled_resource("config.example.toml")
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if source is not None and source.exists():
            shutil.copyfile(source, config_path)
        else:
            config_path.write_text(_FROZEN_FALLBACK_CONFIG, encoding="utf-8")
    except OSError:
        # Last resort: try the minimal seed; if even that fails there is nothing
        # writable, and load_config will use the in-memory defaults.
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(_FROZEN_FALLBACK_CONFIG, encoding="utf-8")
        except OSError:
            pass


def read_common_settings(path: Path) -> CommonSettings:
    cfg = load_config(path)
    return CommonSettings(
        ocr_version=cfg.ocr.ocr_version,
        dpi=cfg.ocr.dpi,
        min_confidence=cfg.ocr.min_confidence,
        device=cfg.ocr.device,
        model_type=cfg.ocr.model_type,
    )


def set_overwrite_original(path: Path, value: bool) -> None:
    """Persist the GUI 覆盖原文件 checkbox to ``[output]`` (keeps comments)."""
    config_path = Path(path)
    ensure_config_file(config_path)

    try:
        doc = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    except TOMLKitError as exc:
        raise ConfigError(f"设置文件 TOML 格式错误: {exc}") from exc

    output_table = _ensure_toml_table(doc, "output")
    output_table["overwrite_original"] = bool(value)
    _write_config_text(config_path, tomlkit.dumps(doc))


def apply_common_settings(path: Path, settings: CommonSettings) -> None:
    config_path = Path(path)
    _validate_common_settings(settings, OcrConfig())

    ensure_config_file(config_path)

    try:
        doc = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    except TOMLKitError as exc:
        raise ConfigError(f"设置文件 TOML 格式错误: {exc}") from exc

    current = load_config(config_path)
    _validate_common_settings(settings, current.ocr)

    ocr_table = _ensure_toml_table(doc, "ocr")

    # 清理 paddle 时代废弃键，保持配置文件整洁。
    for key in _DEPRECATED_OCR_KEYS:
        ocr_table.pop(key, None)

    ocr_table["ocr_version"] = settings.ocr_version
    ocr_table["dpi"] = settings.dpi
    ocr_table["min_confidence"] = settings.min_confidence
    ocr_table["device"] = settings.device
    ocr_table["model_type"] = settings.model_type

    # 0.8 及以前的 [performance] 段已废弃：保存设置时顺带移除。
    doc.pop("performance", None)

    _write_config_text(config_path, tomlkit.dumps(doc))


def _validate_common_settings(
    settings: CommonSettings,
    ocr_base: OcrConfig,
) -> None:
    ocr = replace(
        ocr_base,
        ocr_version=settings.ocr_version,
        dpi=settings.dpi,
        min_confidence=settings.min_confidence,
        device=settings.device,
        model_type=settings.model_type,
    )
    _validate_ocr(ocr)


def _ensure_toml_table(
    doc: MutableMapping[str, object],
    section_name: str,
) -> MutableMapping[str, object]:
    section = doc.get(section_name)
    if section is None:
        section = tomlkit.table()
        doc[section_name] = section
    if not isinstance(section, MutableMapping):
        raise ConfigError(f"设置项 {section_name} 必须是 TOML 表格")
    return section


def _build_section(cls: type[_T], section_name: str, data: dict[str, object]) -> _T:
    allowed = {field.name for field in fields(cls)}
    values: dict[str, object] = {}

    for key, value in data.items():
        if key not in allowed:
            if section_name == "ocr" and key in _DEPRECATED_OCR_KEYS:
                continue
            _warn_unknown(f"{section_name}.{key}")
            continue
        values[key] = value

    return cls(**values)


def _section(values: dict[str, object], key: str, default: _T) -> _T:
    return values.get(key, default)  # type: ignore[return-value]


def _warn_unknown(name: str) -> None:
    warnings.warn(f"未知设置项 {name}，已忽略。", UserWarning, stacklevel=3)


def _validate(cfg: AppConfig) -> AppConfig:
    ocr = _validate_ocr(cfg.ocr)
    output = _validate_output(cfg.output)
    return replace(
        cfg,
        ocr=ocr,
        output=output,
        input=_validate_input(cfg.input),
        gui=_validate_gui(cfg.gui),
        logging=_validate_logging(cfg.logging),
    )


def _validate_ocr(cfg: OcrConfig) -> OcrConfig:
    _require_int("dpi", cfg.dpi)
    if not 72 <= cfg.dpi <= 600:
        _range_error("dpi", "72–600")

    _require_number("min_confidence", cfg.min_confidence)
    if not 0 <= cfg.min_confidence <= 1:
        _range_error("min_confidence", "0–1")

    _require_int("min_existing_chars", cfg.min_existing_chars)
    if cfg.min_existing_chars < 0:
        _range_error("min_existing_chars", "0 以上")

    if cfg.det_limit_side_len is not None:
        _require_int("det_limit_side_len", cfg.det_limit_side_len)
        if cfg.det_limit_side_len <= 0:
            _range_error("det_limit_side_len", "大于 0")

    if not isinstance(cfg.ocr_version, str):
        raise ConfigError("设置项 ocr_version 必须是 PP-OCRv4、PP-OCRv5 或 PP-OCRv6")
    ocr_version = cfg.ocr_version.strip()
    version_aliases = {
        "pp-ocrv4": "PP-OCRv4",
        "pp-ocrv5": "PP-OCRv5",
        "pp-ocrv6": "PP-OCRv6",
        "ppocrv4": "PP-OCRv4",
        "ppocrv5": "PP-OCRv5",
        "ppocrv6": "PP-OCRv6",
    }
    normalized_version = version_aliases.get(ocr_version.casefold().replace("_", "-"))
    if normalized_version is None:
        raise ConfigError("设置项 ocr_version 必须是 PP-OCRv4、PP-OCRv5 或 PP-OCRv6")

    _require_int("cpu_threads", cfg.cpu_threads)
    if not 0 <= cfg.cpu_threads <= 64:
        _range_error("cpu_threads", "0–64")

    _require_bool("textline_orientation", cfg.textline_orientation)

    if not isinstance(cfg.device, str):
        raise ConfigError("设置项 device 必须是 cpu、cuda 或 dml")
    device = cfg.device.casefold()
    if device not in {"cpu", "cuda", "dml"}:
        raise ConfigError("设置项 device 必须是 cpu、cuda 或 dml")

    if not isinstance(cfg.model_type, str):
        raise ConfigError("设置项 model_type 必须是 tiny、small、medium、mobile 或 server")
    model_type = cfg.model_type.casefold()
    # 与 RapidOCR default_models.yaml 一致：v6 只有 tiny/small/medium，
    # v4/v5 只有 mobile/server。非法组合（如旧配置残留 server + v6）
    # 静默归一到该版本的默认模型，保证旧配置平滑迁移。
    if normalized_version == "PP-OCRv6":
        allowed_types = ("tiny", "small", "medium")
        fallback_type = "small"
    else:
        allowed_types = ("mobile", "server")
        fallback_type = "mobile"
    if model_type not in allowed_types:
        model_type = fallback_type

    changes: dict[str, str] = {}
    if model_type != cfg.model_type:
        changes["model_type"] = model_type
    if normalized_version != cfg.ocr_version:
        changes["ocr_version"] = normalized_version
    if device != cfg.device:
        changes["device"] = device
    return replace(cfg, **changes) if changes else cfg


def _validate_output(cfg: OutputConfig) -> OutputConfig:
    _require_bool("overwrite_original", cfg.overwrite_original)
    return cfg


def _validate_input(cfg: InputConfig) -> InputConfig:
    _require_bool("recursive", cfg.recursive)
    image_extensions = _normalize_image_extensions(cfg.image_extensions)
    if image_extensions == cfg.image_extensions:
        return cfg
    return replace(cfg, image_extensions=image_extensions)


def _normalize_image_extensions(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ConfigError("设置项 image_extensions 必须是字符串列表")

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ConfigError("设置项 image_extensions 必须是字符串列表")
        extension = item.removeprefix(".").casefold()
        if extension not in seen:
            result.append(extension)
            seen.add(extension)

    return tuple(result)


def _validate_gui(cfg: GuiConfig) -> GuiConfig:
    if not isinstance(cfg.appearance, str):
        raise ConfigError("设置项 appearance 必须是 system/light/dark")

    appearance = cfg.appearance.casefold()
    if appearance not in {"system", "light", "dark"}:
        raise ConfigError("设置项 appearance 必须是 system/light/dark")

    return cfg if appearance == cfg.appearance else GuiConfig(appearance=appearance)


def _validate_logging(cfg: LoggingConfig) -> LoggingConfig:
    _require_bool("enabled", cfg.enabled)
    if not isinstance(cfg.level, str):
        raise ConfigError("设置项 level 必须是 DEBUG/INFO/WARNING/ERROR")
    if not isinstance(cfg.dir, str):
        raise ConfigError("设置项 dir 必须是字符串")

    level = cfg.level.casefold()
    levels = {
        "debug": "DEBUG",
        "info": "INFO",
        "warning": "WARNING",
        "error": "ERROR",
    }
    if level not in levels:
        raise ConfigError("设置项 level 必须是 DEBUG/INFO/WARNING/ERROR")

    normalized = levels[level]
    return cfg if normalized == cfg.level else replace(cfg, level=normalized)


def _require_int(field_name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"设置项 {field_name} 必须是整数")


def _require_number(field_name: str, value: object) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"设置项 {field_name} 必须是数字")


def _require_bool(field_name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"设置项 {field_name} 必须是布尔值")


def _range_error(field_name: str, expected: str) -> None:
    raise ConfigError(f"设置项 {field_name} 超出范围，应为 {expected}")
