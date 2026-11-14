from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import pytest

tk = pytest.importorskip("tkinter")
ctk = pytest.importorskip("customtkinter")

from pdf_ocrer.gui import App, _paths_from_drop_data, _paths_from_drop_items
from pdf_ocrer.pipeline import BatchSummary, FileResult, FileStatus


@pytest.fixture()
def app(tmp_path: Path):
    try:
        probe = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    else:
        probe.withdraw()
        probe.destroy()

    instance = App(
        config_path=tmp_path / "config.toml",
        engine_factory=lambda cfg: None,  # type: ignore[return-value]
    )
    instance.withdraw()
    try:
        yield instance
    finally:
        instance.destroy()


def test_progress_event_updates_status_table_and_progressbar(app: App) -> None:
    app._queue.put(("progress", 3, 12, 5, 20, "文件.pdf"))
    app._drain_queue()

    status = app.status_var.get()
    assert "个" in status
    assert "页" in status
    assert app.progressbar.get() == pytest.approx((3 - 1 + 5 / 20) / 12)

    rows = app.file_tree.get_children()
    assert len(rows) == 1


def test_ticker_keeps_ocr_status_after_progress(app: App) -> None:
    app._queue.put(("page_start", 1, 1, 1, 2, "两页.pdf"))
    app._drain_queue()
    app._running = True

    app._tick_status()
    try:
        assert "第 1/2 页" in app.status_var.get()
        assert "识别中" in app.status_var.get()
    finally:
        app._cancel_status_timer()


def test_file_done_event_updates_status_table(app: App, tmp_path: Path) -> None:
    result = FileResult(
        source=tmp_path / "原文件.pdf",
        output=tmp_path / "原文件_OCR.pdf",
        status=FileStatus.SUCCESS_OCR,
        total_pages=5,
        ocr_pages=3,
        note="",
    )

    app._queue.put(("file_done", result))
    app._drain_queue()

    rows = app.file_tree.get_children()
    assert len(rows) == 1


def test_drop_data_splits_multiple_unbraced_paths(tmp_path: Path) -> None:
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    first.write_bytes(b"%PDF-1.7\n")
    second.write_bytes(b"%PDF-1.7\n")

    assert _paths_from_drop_data(f"{first} {second}") == [first, second]
    assert _paths_from_drop_data(f"{{{first}}} {second}") == [first, second]


def test_files_event_prefills_rows_as_waiting(app: App) -> None:
    app._queue.put(("files", ["b.pdf", "a.pdf"]))
    app._drain_queue()

    values = [app.file_tree.item(row, "values") for row in app.file_tree.get_children()]
    assert values == [
        ("b.pdf", "等待中", "", ""),
        ("a.pdf", "等待中", "", ""),
    ]


def test_cancelled_summary_marks_unfinished_rows_cancelled(app: App) -> None:
    app._queue.put(("files", ["a.pdf", "b.pdf"]))
    app._queue.put(("progress", 1, 2, 1, 1, "a.pdf"))
    app._drain_queue()

    app._queue.put(("done", BatchSummary(results=[], cancelled=True)))
    app._drain_queue()

    values = [app.file_tree.item(row, "values") for row in app.file_tree.get_children()]
    assert values == [
        ("a.pdf", "已取消", "", ""),
        ("b.pdf", "已取消", "", ""),
    ]


def test_cancelled_done_keeps_progress_bar_position(app: App) -> None:
    app.progressbar.set(0.25)

    app._queue.put(("done", BatchSummary(results=[], cancelled=True)))
    app._drain_queue()

    assert app.progressbar.get() == pytest.approx(0.25)


def test_page_eta_uses_current_file_durations(app: App) -> None:
    app._queue.put(("page_start", 1, 2, 2, 5, "a.pdf"))
    app._drain_queue()
    app._page_durations[1] = [2.0, 4.0]

    assert app._page_eta_seconds() == 9
    app._cancel_status_timer()


def test_status_elapsed_is_file_total_and_persists_between_pages(app: App) -> None:
    import time as _time

    app._queue.put(("page_start", 1, 2, 1, 3, "多页.pdf"))
    app._drain_queue()
    # 回拨文件开始时间，验证“已用”是文件级总耗时
    app._file_started_at[1] = _time.monotonic() - 25
    app.status_var.set(app._status_text())

    page_one = app.status_var.get()
    assert "本文件已用 25 秒" in page_one

    # 第一页完成后（下一页尚未开始），文件级已用时间仍然显示
    app._queue.put(("progress", 1, 2, 1, 3, "多页.pdf"))
    app._drain_queue()
    assert "本文件已用 25 秒" in app.status_var.get()


def test_file_done_stops_elapsed_display(app: App) -> None:
    app._queue.put(("page_start", 1, 1, 1, 1, "单页.pdf"))
    app._drain_queue()
    assert 1 in app._file_started_at

    app._queue.put(("file_done", FileResult(source=Path("单页.pdf"), output=None, status=FileStatus.SUCCESS_OCR, total_pages=1, ocr_pages=1, note="")))
    app._drain_queue()

    assert 1 not in app._file_started_at


def test_overwrite_preference_is_persisted_and_reloaded(tmp_path: Path) -> None:
    from pdf_ocrer.config import load_config, set_overwrite_original

    config_path = tmp_path / "config.toml"
    config_path.write_text("[ocr]\ndpi = 200\n", encoding="utf-8")
    set_overwrite_original(config_path, True)

    assert load_config(config_path).output.overwrite_original is True
    # 再写回 false
    set_overwrite_original(config_path, False)
    assert load_config(config_path).output.overwrite_original is False


def test_drop_data_accepts_files_and_folders(tmp_path: Path) -> None:
    folder = tmp_path / "资 料 夹"
    folder.mkdir()
    pdf = folder / "有 空白.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF-1.7\n")

    paths = _paths_from_drop_items((str(pdf), str(other), str(folder)))
    assert paths == [pdf, other, folder]

    assert _paths_from_drop_data(f"{{{pdf}}}") == [pdf]
    assert _paths_from_drop_data(str(folder)) == [folder]
    assert _paths_from_drop_data(f"{{{pdf}}} {{{other}}}") == [pdf, other]


def test_start_without_paths_shows_error(app: App, monkeypatch) -> None:
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "pdf_ocrer.gui.messagebox.showerror",
        lambda title, message, **kwargs: shown.append((title, message)),
    )

    app._start()

    assert shown and shown[0][0] == "路径错误"
    assert app._worker is None


def test_start_with_missing_path_starts_batch_and_marks_failed(
    app: App, tmp_path: Path, monkeypatch
) -> None:
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "pdf_ocrer.gui.messagebox.showerror",
        lambda title, message, **kwargs: shown.append((title, message)),
    )
    app.paths_var.set(str(tmp_path / "missing.pdf"))

    app._start()
    _wait_for_worker(app)

    # 不存在的路径不再弹窗阻断启动，而是作为失败条目出现在文件列表。
    assert not shown
    rows = [app.file_tree.item(iid, "values") for iid in app.file_tree.get_children()]
    assert rows and rows[0][1] == "失败"


def test_overwrite_checkbox_persists_immediately_on_toggle(app, tmp_path: Path) -> None:
    from pdf_ocrer.config import load_config

    config_path = tmp_path / "config.toml"

    # 勾选即写入，不点“开始处理”也不丢状态。
    app.overwrite_var.set(True)
    app._on_overwrite_toggled()
    assert load_config(config_path).output.overwrite_original is True

    # 取消勾选立即写回 false。
    app.overwrite_var.set(False)
    app._on_overwrite_toggled()
    assert load_config(config_path).output.overwrite_original is False


def test_worker_without_engine_factory_runs_ocr_in_worker_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """生产路径：OCR 在常驻工作子进程执行，避免模型加载饿死 UI 线程的 GIL。"""
    folder = tmp_path / "input"
    folder.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}
    app = App.__new__(App)
    app._config_path = config_path
    app._queue = queue.Queue()
    app._engine_factory = None
    targets = [folder]

    class FakeSession:
        def __init__(self) -> None:
            self.run_calls: list[dict[str, object]] = []
            self.shutdown_calls = 0

        def run_batch(self, run_targets, cfg, engine, **kwargs):
            captured["targets"] = run_targets
            captured["engine"] = engine
            captured["force"] = kwargs.get("force")
            return BatchSummary([], False)

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    session = FakeSession()
    app._session = session

    app._run_worker(targets, threading.Event(), force=True)

    assert captured["targets"] == targets
    assert captured["engine"] is None
    assert captured["force"] is True
    assert app._queue.get_nowait()[0] == "done"


def test_destroy_shuts_down_worker_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """关闭窗口时必须释放常驻 OCR 子进程（后台线程异步执行）。"""
    from customtkinter import CTk

    # 屏蔽 CTk 基类 destroy，避免未初始化的 Tk 对象参与销毁。
    monkeypatch.setattr(CTk, "destroy", lambda self: None)
    app = App.__new__(App)
    app._closed = False
    app._after_id = None
    app._status_after_id = None

    class FakeSession:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    session = FakeSession()
    app._session = session

    app.destroy()

    # shutdown 在后台线程执行：轮询等待，窗口本身不被阻塞。
    deadline = time.monotonic() + 2
    while session.shutdown_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.shutdown_calls == 1
    assert app._session is None


def test_done_event_logs_summary_without_dialog(app: App, tmp_path: Path, monkeypatch) -> None:
    asks: list[str] = []
    monkeypatch.setattr(
        "pdf_ocrer.gui.messagebox.askyesno",
        lambda *args, **kwargs: asks.append(str(args)),
    )
    summary = BatchSummary(
        results=[
            FileResult(
                source=tmp_path / "done.pdf",
                output=tmp_path / "done_OCR.pdf",
                status=FileStatus.SKIPPED_DONE,
                total_pages=0,
                ocr_pages=0,
                note="",
                rel="done.pdf",
            )
        ],
        cancelled=False,
    )
    app._running = True

    app._queue.put(("done", summary))
    app._drain_queue()

    assert asks == []
    assert "跳过：1（加密 0、已输出 1、已OCR 0、无文本 0）" in app.log_text.get("1.0", "end")
    assert "完成：共 1 个文件" in app.status_var.get()


def _wait_for_worker(app: App) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        app.update()
        if app._worker is not None and not app._worker.is_alive():
            app._drain_queue()
            app.update()
            if not app._running:
                return
        time.sleep(0.01)

    raise AssertionError("GUI worker did not finish")


def test_add_input_paths_filters_unsupported_files(app: App, tmp_path: Path) -> None:
    pdf = tmp_path / "报告.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    txt = tmp_path / "说明.txt"
    txt.write_text("hi", encoding="utf-8")
    folder = tmp_path / "文件夹"
    folder.mkdir()

    app._add_input_paths([txt, pdf, folder])

    assert pdf in app._input_paths
    assert folder in app._input_paths
    assert txt not in app._input_paths
    assert "已忽略 1 个" in app.status_var.get()


def test_add_input_paths_rejects_generated_output(app: App, tmp_path: Path) -> None:
    """本工具的输出文件（*_OCR.pdf）拖入时直接忽略，绝不作为输入。"""
    ocr_output = tmp_path / "报告_OCR.pdf"
    ocr_output.write_bytes(b"%PDF-1.7\n")
    pdf = tmp_path / "报告.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    app._add_input_paths([ocr_output, pdf])

    assert pdf in app._input_paths
    assert ocr_output not in app._input_paths
    assert "输出文件" in app.status_var.get()


def test_app_icon_resource_available() -> None:
    """运行时窗口图标资源必须可定位（开发环境=仓库 packaging/app.ico）。"""
    from pdf_ocrer.gui import _locate_app_icon

    icon = _locate_app_icon()
    assert icon is not None
    assert icon.is_file()


def test_set_native_window_icon_smoke(app: App) -> None:
    """Win32 高清图标设置路径在真实窗口上不应抛异常。"""
    from pdf_ocrer.gui import _locate_app_icon, _set_native_window_icon

    icon = _locate_app_icon()
    assert icon is not None

    app.update_idletasks()
    _set_native_window_icon(app, icon)  # 不抛异常即通过（失败路径内部已吞掉）


def test_completion_message_counts_no_text_separately(app: App) -> None:
    """「无文字-跳过」是独立终态，不得归入「成功」计数。"""
    from pdf_ocrer.pipeline import BatchSummary, FileResult

    def make(status: FileStatus, rel: str) -> FileResult:
        return FileResult(
            source=app._input_paths[0] if False else Path(rel),
            output=None,
            status=status,
            total_pages=1,
            ocr_pages=0,
            note="",
            rel=rel,
        )

    summary = BatchSummary(
        results=[
            make(FileStatus.SUCCESS_OCR, "ok.pdf"),
            make(FileStatus.NO_TEXT_FOUND, "blank.pdf"),
            make(FileStatus.NO_TEXT_FOUND, "blank2.pdf"),
            make(FileStatus.SKIPPED_DONE, "done.pdf"),
            make(FileStatus.FAILED, "bad.pdf"),
        ],
        cancelled=False,
    )
    app._running = True
    app._queue.put(("done", summary))
    app._drain_queue()

    logged = "\n".join(app.log_text.get("1.0", "end").splitlines())
    assert "成功：1" in logged
    assert "跳过：3（加密 0、已输出 1、已OCR 0、无文本 2）" in logged
    assert "失败：1" in logged


def test_progress_does_not_overwrite_terminal_row(app: App) -> None:
    """迟到的页事件不得把已完成的行刷回"处理中"。"""
    app._upsert_file_row("文件.pdf", "OCR完成", "文件_OCR.pdf", "3")

    app._queue.put(("progress", 1, 1, 1, 1, "文件.pdf"))
    app._drain_queue()

    values = app.file_tree.item(app.file_tree.get_children()[0], "values")
    assert values[1] == "OCR完成"


def test_rows_display_basename_with_disambiguation_prefix(app: App) -> None:
    """同名消歧前缀保留显示（可区分同名文件），分隔符用 Windows 的 \\。"""
    rels = ["Desktop/OCR验证.pdf", "整车VOC/OCR验证.pdf", "普通.pdf"]
    app._prefill_rows(rels)

    rows = app.file_tree.get_children()
    assert len(rows) == 3  # 行键唯一：三个 rel 三行
    names = [app.file_tree.item(iid, "values")[0] for iid in rows]
    assert names == ["Desktop\\OCR验证.pdf", "整车VOC\\OCR验证.pdf", "普通.pdf"]


def test_sync_paths_entry_scrolls_view_to_end(app: App, monkeypatch) -> None:
    calls: list[float] = []
    monkeypatch.setattr(
        app.paths_entry, "xview_moveto", lambda fraction: calls.append(fraction)
    )

    app._sync_paths_entry()

    assert calls and calls[-1] == 1.0


def test_tree_tooltip_hidden_when_not_hovering(app: App) -> None:
    app._hide_tree_tooltip()  # 从未显示时安全无副作用
    assert app._tree_tooltip is None

    app._show_tree_tooltip("很长很长的文件名示例.pdf", 100, 100)
    assert app._tree_tooltip is not None
    app._hide_tree_tooltip()
    assert not app._tree_tooltip.winfo_viewable()


def test_picker_menu_font_and_row_height(app: App) -> None:
    """「选择文件/文件夹」弹出菜单：字体与 CTk 渲染一致、行高随缩放折算。"""
    import re

    def font_px(raw: str) -> int:
        match = re.search(r"(-\d+)", raw)
        assert match is not None, f"font value has no negative pixel size: {raw}"
        return int(match.group(1))

    menu = app._build_picker_menu()
    try:
        items = [child for child in menu.winfo_children()]
        assert len(items) == 2

        label = ctk.CTkLabel(app, text="基准", font=ctk.CTkFont(size=14))
        try:
            expected_px = font_px(str(label._label.cget("font")))
            scaling = ctk.ScalingTracker.get_widget_scaling(app)
            min_height = round(30 * scaling) - 1
            for item in items:
                assert font_px(str(item.cget("font"))) == expected_px
                # 行高 = 字体高 + 上下留白，随缩放同步增长。
                assert item.winfo_reqheight() >= min_height
        finally:
            label.destroy()
    finally:
        menu.destroy()
