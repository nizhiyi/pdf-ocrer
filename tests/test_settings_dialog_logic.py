from __future__ import annotations

import pytest

from pdf_ocrer.settings_dialog import (
    DEFAULT_MODEL_TYPE_BY_VERSION,
    MODEL_TYPES_BY_VERSION,
    OCR_VERSIONS,
    model_types_for_version,
    resolve_model_type,
)


def test_ocr_versions_cover_all_supported() -> None:
    assert OCR_VERSIONS == ("PP-OCRv4", "PP-OCRv5", "PP-OCRv6")


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("PP-OCRv4", ("mobile", "server")),
        ("PP-OCRv5", ("mobile", "server")),
        ("PP-OCRv6", ("tiny", "small", "medium")),
    ],
)
def test_model_types_for_version_matches_rapidocr_matrix(version, expected):
    assert model_types_for_version(version) == expected
    assert MODEL_TYPES_BY_VERSION[version] == expected


def test_v6_does_not_support_mobile_or_server() -> None:
    # PP-OCRv6 只有 tiny/small/medium；mobile/server 是 v4/v5 专属。
    assert "mobile" not in model_types_for_version("PP-OCRv6")
    assert "server" not in model_types_for_version("PP-OCRv6")
    assert "tiny" not in model_types_for_version("PP-OCRv4")


@pytest.mark.parametrize(
    ("version", "model_type", "expected"),
    [
        ("PP-OCRv6", "small", "small"),
        ("PP-OCRv6", "tiny", "tiny"),
        ("PP-OCRv6", "medium", "medium"),
        ("PP-OCRv6", "server", "small"),  # v6 无 server → 回退 v6 默认
        ("PP-OCRv6", "mobile", "small"),
        ("PP-OCRv4", "mobile", "mobile"),
        ("PP-OCRv4", "server", "server"),
        ("PP-OCRv4", "small", "mobile"),  # v4 无 small → 回退 v4 默认
        ("PP-OCRv5", "tiny", "mobile"),
        ("PP-OCRv4", "unknown", "mobile"),
    ],
)
def test_resolve_model_type_falls_back_to_version_default(version, model_type, expected):
    assert resolve_model_type(version, model_type) == expected


def test_defaults_are_consistent_with_allowed_types() -> None:
    for version, default in DEFAULT_MODEL_TYPE_BY_VERSION.items():
        assert default in model_types_for_version(version)
