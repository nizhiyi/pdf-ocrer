from __future__ import annotations

from pathlib import Path

from pdf_ocrer.config import InputConfig
from pdf_ocrer.scanning import scan_inputs


def test_scan_inputs_non_recursive_top_level_pdfs_sorted(tmp_path: Path) -> None:
    _write(tmp_path / "B.PDF")
    _write(tmp_path / "a.pdf")
    _write(tmp_path / "note.txt")
    _write(tmp_path / "sub" / "nested.pdf")

    items = scan_inputs(tmp_path, InputConfig())

    assert [(item.src, item.rel) for item in items] == [
        (tmp_path / "a.pdf", "a.pdf"),
        (tmp_path / "B.PDF", "B.PDF"),
    ]


def test_scan_inputs_excludes_generated_output_and_temp_files(tmp_path: Path) -> None:
    _write(tmp_path / "doc.pdf")
    _write(tmp_path / "doc_OCR.pdf")
    _write(tmp_path / "doc_ocr.PDF")
    _write(tmp_path / "~temp123.ocrtmp.pdf")
    _write(tmp_path / "OCR_notes.pdf")

    items = scan_inputs(tmp_path, InputConfig())

    assert [item.rel for item in items] == ["doc.pdf", "OCR_notes.pdf"]


def test_scan_inputs_recursive_calculates_posix_rel(tmp_path: Path) -> None:
    _write(tmp_path / "top.pdf")
    _write(tmp_path / "sub" / "B.pdf")
    _write(tmp_path / "sub" / "deep" / "a.PDF")
    _write(tmp_path / "sub" / "deep" / "note.txt")
    _write(tmp_path / "sub" / "deep" / "a_OCR.pdf")

    items = scan_inputs(tmp_path, InputConfig(recursive=True))

    assert [item.rel for item in items] == [
        "sub/B.pdf",
        "sub/deep/a.PDF",
        "top.pdf",
    ]
    assert [item.src for item in items] == [
        tmp_path / "sub" / "B.pdf",
        tmp_path / "sub" / "deep" / "a.PDF",
        tmp_path / "top.pdf",
    ]


def test_scan_inputs_includes_pdf_and_configured_images(tmp_path: Path) -> None:
    _write(tmp_path / "doc.pdf")
    _write(tmp_path / "photo.JPG")
    _write(tmp_path / "scan.png")
    _write(tmp_path / "pages.TIFF")
    _write(tmp_path / "note.txt")

    items = scan_inputs(tmp_path, InputConfig())

    assert [item.rel for item in items] == [
        "doc.pdf",
        "pages.TIFF",
        "photo.JPG",
        "scan.png",
    ]


def test_scan_inputs_empty_image_extensions_only_includes_pdfs(tmp_path: Path) -> None:
    _write(tmp_path / "doc.pdf")
    _write(tmp_path / "photo.jpg")
    _write(tmp_path / "scan.png")

    items = scan_inputs(tmp_path, InputConfig(image_extensions=()))

    assert [item.rel for item in items] == ["doc.pdf"]


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\n")
