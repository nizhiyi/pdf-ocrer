from __future__ import annotations

import sys
import warnings
from dataclasses import replace

import pytest

from pdf_ocrer.config import (
    CommonSettings,
    ConfigError,
    InputConfig,
    LoggingConfig,
    apply_common_settings,
    ensure_config_file,
    load_config,
    read_common_settings,
)


def _common_settings(**overrides: object) -> CommonSettings:
    values = {
        "ocr_version": "PP-OCRv6",
        "dpi": 300,
        "min_confidence": 0.75,
        "device": "cpu",
        "model_type": "small",
    }
    values.update(overrides)
    return CommonSettings(**values)


def test_common_settings_roundtrip_device_and_model_type(tmp_path):
    config_path = tmp_path / "config.toml"
    settings = _common_settings(device="dml", model_type="medium")

    apply_common_settings(config_path, settings)

    assert read_common_settings(config_path) == settings
    cfg = load_config(config_path)
    assert cfg.ocr.device == "dml"
    assert cfg.ocr.model_type == "medium"


def test_apply_common_settings_rejects_invalid_device(tmp_path):
    config_path = tmp_path / "config.toml"

    with pytest.raises(ConfigError, match="device"):
        apply_common_settings(config_path, _common_settings(device="gpu"))


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "nope.toml")

    assert cfg.ocr.dpi == 200
    assert cfg.ocr.ocr_version == "PP-OCRv6"
    assert cfg.ocr.model_type == "small"
    assert cfg.ocr.cpu_threads == 0
    assert cfg.ocr.textline_orientation is True
    assert cfg.output.overwrite_original is False
    assert cfg.input == InputConfig()
    assert cfg.gui.appearance == "system"
    assert cfg.logging == LoggingConfig()


def test_deprecated_performance_section_is_silently_ignored(tmp_path):
    # 0.8 及以前的 [performance] 段已废弃：加载时静默忽略，不告警不报错。
    p = tmp_path / "c.toml"
    p.write_text("[performance]\nworkers = 4\n\n[ocr]\ndpi = 200\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config(p)

    assert cfg.ocr.dpi == 200


def test_output_overwrite_original_loads_from_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[output]\noverwrite_original = true\n", encoding="utf-8")

    assert load_config(p).output.overwrite_original is True


def test_set_overwrite_original_persists_and_keeps_comments(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("# my config\n[output]\n# keep me\n\n[ocr]\ndpi = 250\n", encoding="utf-8")

    from pdf_ocrer.config import set_overwrite_original

    set_overwrite_original(p, True)

    text = p.read_text(encoding="utf-8")
    assert "# my config" in text
    assert "# keep me" in text
    assert "overwrite_original = true" in text
    assert load_config(p).output.overwrite_original is True
    set_overwrite_original(p, False)
    assert load_config(p).output.overwrite_original is False


def test_deprecated_sections_are_ignored_silently(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        "[naming]\n"
        "enabled = true\n"
        "\n"
        "[llm]\n"
        "provider = \"openai_compatible\"\n"
        "\n"
        "[watch]\n"
        "poll_seconds = 1.0\n"
        "\n"
        "[ocr]\n"
        "dpi = 250\n",
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)

    assert cfg.ocr.dpi == 250
    assert not hasattr(cfg, "naming")
    assert not hasattr(cfg, "llm")
    assert not hasattr(cfg, "watch")
    assert caught == []


def test_input_image_extensions_defaults() -> None:
    assert InputConfig().image_extensions == ("jpg", "jpeg", "png", "tif", "tiff")


def test_invalid_input_recursive_type_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[input]\nrecursive = "yes"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="recursive"):
        load_config(p)


def test_invalid_output_overwrite_original_type_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[output]\noverwrite_original = "yes"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="overwrite_original"):
        load_config(p)


@pytest.mark.parametrize(
    "key",
    ["engine", "lang", "enable_mkldnn", "det_model_name", "rec_model_name"],
)
def test_deprecated_ocr_keys_are_silently_ignored(tmp_path, key, recwarn):
    p = tmp_path / "c.toml"
    p.write_text(f'[ocr]\n{key} = "whatever"\ndpi = 250\n', encoding="utf-8")

    cfg = load_config(p)

    assert cfg.ocr.dpi == 250
    assert [w for w in recwarn.list if key in str(w.message)] == []


@pytest.mark.parametrize("cpu_threads", [-1, 65])
def test_invalid_cpu_threads_range_raises(tmp_path, cpu_threads):
    p = tmp_path / "c.toml"
    p.write_text(f"[ocr]\ncpu_threads = {cpu_threads}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="cpu_threads"):
        load_config(p)


@pytest.mark.parametrize("bad", ["gpu", "npu", "gpu:0", "gpU"])
def test_invalid_ocr_device_raises(tmp_path, bad):
    p = tmp_path / "c.toml"
    p.write_text(f'[ocr]\ndevice = "{bad}"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="device.*cpu.*cuda.*dml"):
        load_config(p)


@pytest.mark.parametrize("device", ["cuda", "dml"])
def test_gpu_device_allowed(tmp_path, device):
    p = tmp_path / "c.toml"
    p.write_text(f'[ocr]\ndevice = "{device}"\n', encoding="utf-8")

    assert load_config(p).ocr.device == device


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("small", "small"), ("tiny", "tiny"), ("Medium", "medium"), ("SMALL", "small")],
)
def test_ocr_model_type_loads_and_normalizes(tmp_path, raw, expected):
    p = tmp_path / "c.toml"
    p.write_text(f'[ocr]\nmodel_type = "{raw}"\n', encoding="utf-8")

    assert load_config(p).ocr.model_type == expected


@pytest.mark.parametrize(
    ("version", "raw", "expected"),
    [
        ("PP-OCRv6", "server", "small"),
        ("PP-OCRv6", "mobile", "small"),
        ("PP-OCRv6", "large", "small"),
        ("PP-OCRv6", "", "small"),
        ("PP-OCRv4", "small", "mobile"),
        ("PP-OCRv4", "tiny", "mobile"),
        ("PP-OCRv5", "medium", "mobile"),
    ],
)
def test_invalid_model_type_falls_back_to_version_default(tmp_path, version, raw, expected):
    p = tmp_path / "c.toml"
    p.write_text(f'[ocr]\nocr_version = "{version}"\nmodel_type = "{raw}"\n', encoding="utf-8")

    cfg = load_config(p)

    assert cfg.ocr.model_type == expected
    assert cfg.ocr.ocr_version == version


def test_non_string_model_type_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[ocr]\nmodel_type = 5\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="model_type"):
        load_config(p)


@pytest.mark.parametrize("bad", ["PP-OCRv3", "v6", "", "PP-OCRv7"])
def test_invalid_ocr_version_raises(tmp_path, bad):
    p = tmp_path / "c.toml"
    p.write_text(f'[ocr]\nocr_version = "{bad}"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="ocr_version"):
        load_config(p)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("PP-OCRv4", "PP-OCRv4"), ("pp-ocrv5", "PP-OCRv5"), ("ppocrv6", "PP-OCRv6")],
)
def test_ocr_version_loads_and_normalizes(tmp_path, raw, expected):
    p = tmp_path / "c.toml"
    p.write_text(f'[ocr]\nocr_version = "{raw}"\n', encoding="utf-8")

    assert load_config(p).ocr.ocr_version == expected


def test_default_config_path_is_per_user_when_frozen(monkeypatch, tmp_path):
    from pdf_ocrer.config import default_config_path

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path))

    assert default_config_path() == tmp_path / "pdf_ocrer" / "config.toml"


def test_bootstrap_frozen_config_seeds_from_bundle(monkeypatch, tmp_path):
    from pdf_ocrer import config as config_mod

    meipass = tmp_path / "_internal"
    meipass.mkdir()
    (meipass / "config.installer.toml").write_text('[ocr]\n', encoding="utf-8")
    target = tmp_path / "appdata" / "pdf_ocrer" / "config.toml"

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)

    config_mod.bootstrap_frozen_config(target)

    assert target.exists()
    assert load_config(target).ocr.ocr_version == "PP-OCRv6"


def test_bootstrap_frozen_config_falls_back_to_rapidocr_when_copy_fails(monkeypatch, tmp_path):
    from pdf_ocrer import config as config_mod

    meipass = tmp_path / "_internal"
    meipass.mkdir()
    (meipass / "config.installer.toml").write_text('[ocr]\n', encoding="utf-8")
    target = tmp_path / "appdata" / "pdf_ocrer" / "config.toml"

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(config_mod.shutil, "copyfile", boom)

    config_mod.bootstrap_frozen_config(target)

    # Must never leave the frozen app to fall back to the paddle default engine,
    # which the packaged build does not ship.
    assert target.exists()
    assert load_config(target).ocr.ocr_version == "PP-OCRv6"


def test_frozen_missing_config_uses_default_models(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    # Even if bootstrap never ran / could not write anything, a frozen build must
    # land on the bundled PP-OCRv6 small defaults.
    cfg = load_config(tmp_path / "does_not_exist.toml")

    assert cfg.ocr.ocr_version == "PP-OCRv6"
    assert cfg.ocr.model_type == "small"


def test_bootstrap_frozen_config_does_not_overwrite_existing(monkeypatch, tmp_path):
    from pdf_ocrer import config as config_mod

    meipass = tmp_path / "_internal"
    meipass.mkdir()
    (meipass / "config.installer.toml").write_text('[ocr]\n', encoding="utf-8")
    target = tmp_path / "config.toml"
    target.write_text('[ocr]\ndpi = 250\n', encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)

    config_mod.bootstrap_frozen_config(target)

    assert load_config(target).ocr.dpi == 250  # existing user config preserved


def test_invalid_logging_level_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[logging]\nlevel = "TRACE"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="level"):
        load_config(p)


def test_invalid_gui_appearance_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[gui]\nappearance = \"blue\"\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="appearance"):
        load_config(p)


def test_unknown_key_warns(tmp_path, recwarn):
    p = tmp_path / "c.toml"
    p.write_text("[ocr]\nbogus = 1\n", encoding="utf-8")

    load_config(p)

    assert any("bogus" in str(w.message) for w in recwarn.list)


def test_invalid_dpi_raises(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[ocr]\ndpi = 10\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="dpi"):
        load_config(p)


def test_ensure_config_file_creates_empty_file_without_example(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "nested" / "config.toml"

    ensure_config_file(config_path)

    assert config_path.read_text(encoding="utf-8") == ""


def test_ensure_config_file_copies_example_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    example = tmp_path / "config.example.toml"
    example.write_text("# template\n[ocr]\ndpi = 222\n", encoding="utf-8")
    config_path = tmp_path / "config" / "config.toml"

    ensure_config_file(config_path)

    assert config_path.read_text(encoding="utf-8") == example.read_text(encoding="utf-8")


def test_apply_common_settings_preserves_comments_and_removes_deprecated_keys(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "# root comment\n"
        "\n"
        "[ocr]\n"
        "# dpi comment\n"
        "engine = \"paddle\"\n"
        "dpi = 200\n"
        "lang = \"chinese_cht\"\n"
        "min_confidence = 0.5\n"
        "det_model_name = \"old_det\"\n"
        "rec_model_name = \"old_rec\"\n"
        "\n"
        "[naming]\n"
        "enabled = true\n"
        "\n"
        "[performance]\n"
        "workers = 1\n"
        "\n"
        "[llm]\n"
        "provider = \"openai_compatible\"\n"
        "\n"
        "[debug]\n"
        "visible_text = true\n",
        encoding="utf-8",
    )

    settings = _common_settings()
    apply_common_settings(config_path, settings)

    text = config_path.read_text(encoding="utf-8")
    cfg = load_config(config_path)
    assert "# root comment" in text
    assert "# dpi comment" in text
    assert cfg.ocr.ocr_version == "PP-OCRv6"
    assert cfg.ocr.dpi == 300
    assert cfg.ocr.min_confidence == 0.75
    assert cfg.debug.visible_text is True
    # naming/llm 段已废弃：既不写入也不删除，加载时静默忽略。
    assert "[naming]" in text
    assert "[llm]" in text
    # performance 段已废弃：保存设置时自动移除。
    assert "[performance]" not in text
    assert "workers" not in text
    # paddle 时代废弃键在保存时清理。
    assert "engine = " not in text
    assert "lang = " not in text
    assert "det_model_name" not in text
    assert "rec_model_name" not in text
    assert read_common_settings(config_path) == settings


def test_apply_common_settings_writes_ocr_version(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[ocr]\ndpi = 200\n", encoding="utf-8")

    apply_common_settings(config_path, _common_settings(ocr_version="PP-OCRv4", model_type="mobile"))

    text = config_path.read_text(encoding="utf-8")
    assert "ocr_version = \"PP-OCRv4\"" in text
    assert load_config(config_path).ocr.ocr_version == "PP-OCRv4"


def test_apply_common_settings_does_not_write_when_validation_fails(tmp_path):
    config_path = tmp_path / "config.toml"
    original = "[ocr]\ndpi = 200\n"
    config_path.write_text(original, encoding="utf-8")
    original_bytes = config_path.read_bytes()

    with pytest.raises(ConfigError, match="dpi"):
        apply_common_settings(config_path, replace(_common_settings(), dpi=9999))

    assert config_path.read_bytes() == original_bytes


def test_apply_common_settings_does_not_write_when_ocr_version_validation_fails(tmp_path):
    config_path = tmp_path / "config.toml"
    original = "[ocr]\ndpi = 200\n"
    config_path.write_text(original, encoding="utf-8")
    original_bytes = config_path.read_bytes()

    with pytest.raises(ConfigError, match="ocr_version"):
        apply_common_settings(config_path, replace(_common_settings(), ocr_version="PP-OCRv9"))

    assert config_path.read_bytes() == original_bytes


def test_apply_common_settings_bootstraps_missing_file_from_example(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    example = tmp_path / "config.example.toml"
    example.write_text(
        "# template comment\n"
        "[ocr]\n"
        "dpi = 200\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "nested" / "config.toml"

    apply_common_settings(config_path, _common_settings())

    text = config_path.read_text(encoding="utf-8")
    cfg = load_config(config_path)
    assert "# template comment" in text
    assert cfg.ocr.ocr_version == "PP-OCRv6"
    assert cfg.ocr.dpi == 300
