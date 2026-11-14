# Contributing to pdf-ocrer

Thanks for your interest! This document explains how to set up a development
environment and the conventions this project follows.

## Development setup

```bash
git clone https://github.com/nizhiyi/pdf-ocrer.git
cd pdf-ocrer
python -m venv .venv               # Python 3.11+ (3.12 recommended)
.venv\Scripts\activate             # Windows
pip install -e .[dev]              # RapidOCR + ONNX Runtime (bundled, no download)
```

Contributors who only work on non-OCR modules (config, packaging, tests) can
skip the heavy OCR stack:

```bash
pip install -e .[dev] pymupdf
```

Unit tests never import RapidOCR, so they run fine without it.

## Test layers

| Layer | Command | Requirements |
|---|---|---|
| Unit (default) | `pytest` | no network, no OCR models, no GUI |
| Integration | `pytest -o python_files="integration_smoke.py" tests/integration_smoke.py` | RapidOCR models (bundled with the wheel) |

Rules for unit tests:

- Never touch the network.
- Never import `rapidocr` / `onnxruntime` / `pymupdf` at module level in `src/`
  (those imports stay lazy for this reason).
- Never open a Tk window (GUI tests only construct widgets when a display is
  available and must skip otherwise).

## Code style

- `ruff check .` must pass (line length 100, target py311).
- Type hints on all public functions.
- Docstrings and code comments in English; user-facing strings in
  Simplified Chinese (this tool ships with a zh-CN UI).

## Pull requests

- Write tests first (this repo is developed TDD-style); every behavior change
  needs a test.
- Keep PRs focused on one concern.
- Run `pytest` and `ruff check .` before pushing.
