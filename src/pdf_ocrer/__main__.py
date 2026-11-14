import multiprocessing

from pdf_ocrer.cli import main

if __name__ == "__main__":
    # Required before the OCR worker process spawn in a frozen build, or the
    # child would recursively relaunch the app. Must be the first call.
    multiprocessing.freeze_support()
    raise SystemExit(main())
