from __future__ import annotations

import pymupdf
import pytest
from fixtures_gen import GT_LINES, build_all, build_image_jpg, build_image_png


def test_build_all_creates_expected_fixture_files(tmp_path):
    paths = build_all(tmp_path)

    assert set(paths) == {
        "native",
        "scanned",
        "rotated",
        "rotated_180",
        "rotated_270",
        "mixed",
        "encrypted",
        "corrupt",
    }
    for name, path in paths.items():
        assert path == tmp_path / f"{name}.pdf"


def test_native_has_text(fixtures_dir):
    text = pymupdf.open(fixtures_dir / "native.pdf")[0].get_text()

    assert "诊断证明书" in text
    for _, _, expected in GT_LINES:
        assert expected in text


def test_encrypted_needs_pass(fixtures_dir):
    doc = pymupdf.open(fixtures_dir / "encrypted.pdf")

    assert doc.needs_pass
    assert doc.authenticate("test") > 0


def test_corrupt_file_is_not_openable(fixtures_dir):
    with pytest.raises(pymupdf.FileDataError):
        pymupdf.open(fixtures_dir / "corrupt.pdf")


@pytest.mark.parametrize(
    ("builder", "filename"),
    [(build_image_png, "scan.png"), (build_image_jpg, "scan.jpg")],
)
def test_image_fixtures_convert_to_single_page_pdf(tmp_path, builder, filename: str):
    path = tmp_path / filename
    builder(path)

    image_doc = pymupdf.open(path)
    pdf_doc = pymupdf.open("pdf", image_doc.convert_to_pdf())

    assert pdf_doc.page_count == 1
    assert abs(pdf_doc[0].rect.width - 595) < 2
    assert abs(pdf_doc[0].rect.height - 842) < 2


