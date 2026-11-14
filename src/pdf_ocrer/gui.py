from __future__ import annotations

import ctypes
import logging
import queue
import re
import sys
import threading
import time
import tkinter as tk
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import customtkinter as ctk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # 未安装 tkinterdnd2 或原生 tkdnd DLL 缺失：禁用拖放即可
    DND_FILES = None
    TkinterDnD = None
    _DND_AVAILABLE = False
else:
    _DND_AVAILABLE = True

from pdf_ocrer import __version__, ui_theme
from pdf_ocrer.app_logging import setup_logging
from pdf_ocrer.config import (
    ConfigError,
    GuiConfig,
    LoggingConfig,
    OcrConfig,
    bootstrap_frozen_config,
    default_config_path,
    load_config,
    set_overwrite_original,
)
from pdf_ocrer.ocr_engine import OcrEngineProtocol, create_engine
from pdf_ocrer.pipeline import BatchSummary, FileResult, FileStatus, run_batch
from pdf_ocrer.runner import WorkerSession
from pdf_ocrer.scanning import is_generated_output
from pdf_ocrer.settings_dialog import SettingsDialog

EngineFactory = Callable[[OcrConfig], OcrEngineProtocol]
GuiEvent = tuple[object, ...]

_TREE_COLUMNS = ("source", "status", "output", "ocr_pages")


def _locate_app_icon() -> Path | None:
    """运行时窗口图标（packaging/app.ico，与安装包图标同一设计）。"""
    if getattr(sys, "frozen", False):
        candidates: list[Path] = []
        base = getattr(sys, "_MEIPASS", None)
        if base:
            candidates.append(Path(base) / "app.ico")
        candidates.append(Path(sys.executable).resolve().parent / "app.ico")
    else:
        candidates = [Path(__file__).resolve().parents[2] / "packaging" / "app.ico"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _set_native_window_icon(window: tk.Misc, icon_path: Path) -> None:
    """按当前 DPI 直接加载 .ico 对应尺寸的帧并设置窗口图标。

    Tk 的 ``iconbitmap`` 只取 ico 里的单个小帧再拉伸，高分辨率下模糊；
    这里用 WM_SETICON 分别设置标题栏小图标与任务栏大图标，各自取
    与系统 DPI 匹配的清晰尺寸。
    """
    if sys.platform != "win32":
        return
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(window.winfo_id()) or window.winfo_id()
        dpi = user32.GetDpiForWindow(hwnd) or 96
        small = max(16, round(16 * dpi / 96))
        big = max(32, round(32 * dpi / 96))
        image_icon, load_from_file = 1, 0x10

        def load(px: int) -> int:
            return user32.LoadImageW(None, str(icon_path), image_icon, px, px, load_from_file)

        wm_seticon = 0x80
        user32.SendMessageW(hwnd, wm_seticon, 0, load(small))  # ICON_SMALL
        user32.SendMessageW(hwnd, wm_seticon, 1, load(big))  # ICON_BIG
    except (OSError, AttributeError, tk.TclError):
        # 图标美化失败不影响功能；Tk iconbitmap 已提供基础效果。
        pass

# 拖入/选择文件的扩展名白名单（与文件对话框的格式列表一致）：
# 仅接受 PDF 和支持的图片，txt 等其它文件拖入时直接忽略。
_DROP_ALLOWED_EXTENSIONS = frozenset({"pdf", "jpg", "jpeg", "png", "tif", "tiff"})
_PICK_FILE_LABEL = "选择 PDF 文件…（可多选）"
_PICK_FOLDER_LABEL = "选择文件夹…"
_logger = logging.getLogger(__name__)


def run_gui(config_path: Path | None = None) -> None:
    app = App(config_path=config_path)
    app.mainloop()


def _effective_appearance(mode: str) -> str:
    normalized = mode.casefold()
    if normalized == "system":
        getter = getattr(ctk, "get_appearance_mode", None)
        if callable(getter):
            current = str(getter()).casefold()
            if current in {"light", "dark"}:
                return current
        return "light"
    return "dark" if normalized == "dark" else "light"


def _style_treeview(mode: str, widget: tk.Misc | None = None) -> None:
    appearance = _effective_appearance(mode)
    colors = ui_theme.palette(appearance)

    style = ttk.Style()
    # ttk 字体的「点」按 dpi/72 折算，CTk 字体按 dpi/96 折算，同样字号差 4/3；
    # Treeview 是界面里唯一的 ttk 控件，字体改用固定像素（负值）并按 CTk
    # 同源的 widget scaling 换算，与界面其它 CTkFont 文字保持同一像素大小。
    if widget is not None:
        scaling = float(ctk.ScalingTracker.get_widget_scaling(widget))
    else:
        scaling = 1.0
    if "clam" in style.theme_names():
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
    style.configure(
        "PdfOcrer.Treeview",
        background=colors["tree_bg"],
        fieldbackground=colors["tree_bg"],
        foreground=colors["text"],
        borderwidth=0,
        relief="flat",
        bordercolor=colors["card_border"],
        lightcolor=colors["tree_bg"],
        darkcolor=colors["tree_bg"],
        # 负数 = 固定像素：像素值已按 CTk scaling 折算，不再被 tk scaling 二次放大。
        rowheight=max(32, int(14 * scaling + 14)),
        font=(ui_theme.FONT_FAMILY, -round(14 * scaling)),
    )
    style.map(
        "PdfOcrer.Treeview",
        background=[("selected", colors["select"])],
        foreground=[("selected", colors["select_text"])],
    )
    style.configure(
        "PdfOcrer.Treeview.Heading",
        background=colors["heading"],
        foreground=colors["muted"],
        borderwidth=0,
        relief="flat",
        font=(ui_theme.FONT_FAMILY, -round(14 * scaling), "bold"),
        # 表头默认 padding 只有几像素固定值，文字放大后上下贴边；
        # 按 CTk scaling 加留白，保证表头高度与数据行节奏一致。
        padding=(max(8, int(8 * scaling)), max(7, int(7 * scaling))),
    )
    style.map("PdfOcrer.Treeview.Heading", relief=[("active", "flat")])


_TCL_INTERP: tk.Tcl | None = None


def _drop_splitlist(data: str) -> tuple[str, ...]:
    """按 Tcl 列表规则拆分拖放数据（tkdnd 对含空格路径加花括号）。"""
    global _TCL_INTERP
    if _TCL_INTERP is None:
        _TCL_INTERP = tk.Tcl()
    return tuple(_TCL_INTERP.splitlist(data))


def _paths_from_drop_data(data: str) -> list[Path]:
    stripped = data.strip()
    if not stripped:
        return []

    candidates: list[list[Path]] = []
    try:
        candidates.append(_paths_from_drop_items(_drop_splitlist(stripped)))
    except tk.TclError:
        pass
    # Tcl 列表解析会把未加花括号的反斜杠路径（如 C:\a\b.pdf）里的
    # 反斜杠当转义符，解析失败时再用正则兜底，保证多路径仍能被拆分。
    candidates.append(_paths_from_drop_items(_fallback_split_drop_data(stripped)))

    best: list[Path] = []
    for paths in candidates:
        if len(paths) > len(best):
            best = paths
    if best:
        return best

    # 兼容"单个含空格路径、未加花括号"的旧格式。
    whole = Path(stripped)
    if whole.is_file() or whole.is_dir():
        return [whole]
    return []


def _fallback_split_drop_data(data: str) -> tuple[str, ...]:
    items: list[str] = []
    for braced, bare in re.findall(r"\{([^}]*)\}|(\S+)", data):
        value = braced or bare
        if value and value not in items:
            items.append(value)
    return tuple(items)


def _paths_from_drop_items(items: tuple[str, ...] | list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in items:
        path = Path(str(raw))
        if path.is_file() or path.is_dir():
            paths.append(path)
    return paths


if _DND_AVAILABLE:

    class _AppBase(ctk.CTk, TkinterDnD.DnDWrapper):  # type: ignore[union-attr]
        pass

else:

    class _AppBase(ctk.CTk):
        pass


class App(_AppBase):
    def __init__(
        self,
        config_path: Path | None = None,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        super().__init__()
        self._config_path = (
            default_config_path() if config_path is None else Path(config_path)
        )
        bootstrap_frozen_config(self._config_path)
        self._engine_factory = engine_factory
        # 常驻 OCR 工作子进程：首个批次懒启动，模型加载一次直到程序关闭。
        # 构造无副作用（不启动进程），测试注入 engine_factory 时不会被用到。
        self._session = WorkerSession()
        self._queue: queue.Queue[GuiEvent] = queue.Queue()
        self._cancel_event: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._running = False
        self._closed = False
        self._after_id: str | None = None
        self._file_rows: dict[str, str] = {}
        self._tree_tooltip: tk.Toplevel | None = None
        self._tree_tooltip_label: tk.Label | None = None
        self._tree_tooltip_key: tuple[str, str, str] | None = None
        self._tree_hover_font: tkfont.Font | None = None
        self._dnd_enabled = False
        self._input_paths: list[Path] = []
        self._picker_menu: tk.Toplevel | None = None
        self._close_picker_menu: Callable[..., None] | None = None
        self._run_started_at: float | None = None
        self._progress_seen = False
        self._status_after_id: str | None = None
        self._page_started_at: dict[int, float] = {}
        self._page_durations: dict[int, list[float]] = {}
        self._file_started_at: dict[int, float] = {}
        self._current_file_i = 0
        self._current_file_n = 0
        self._current_page_i = 0
        self._current_page_n = 0
        self._current_name = ""

        gui_config = self._load_initial_gui_config()
        initial_appearance = gui_config.appearance
        ctk.set_appearance_mode(initial_appearance)

        self.paths_var = tk.StringVar()
        self.status_var = tk.StringVar(value="选择文件（夹）后开始")
        self.force_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=self._load_initial_overwrite_pref())

        self.title(f"pdf-ocrer {__version__}")
        # 窗口/任务栏图标：替换 CustomTkinter 的默认圆环 logo。
        # iconbitmap 提供基础效果；Win32 WM_SETICON 按 DPI 加载清晰帧。
        icon_path = _locate_app_icon()
        if icon_path is not None:
            try:
                self.iconbitmap(str(icon_path))
            except tk.TclError:
                pass
            self.after(50, lambda: _set_native_window_icon(self, icon_path))
        self.minsize(940, 600)
        self._apply_default_geometry(1000, 700)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_widgets(initial_appearance)
        self._set_running(False)
        self._setup_drag_drop()
        if not self._dnd_enabled:
            self._append_log("拖放功能不可用")
        self._after_id = self.after(100, self._poll_queue)

    def _apply_default_geometry(self, width: int, height: int) -> None:
        """宽而矮的默认窗口，并在屏幕内居中。"""
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2 - 20)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def destroy(self) -> None:
        self._closed = True
        self._cancel_status_timer()
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        # 释放常驻 OCR 子进程：放在后台线程执行，窗口立即关闭不"慢一拍"。
        # 异常退出路径由子进程的父进程看门狗兜底，临时文件由下次批次清理。
        session = getattr(self, "_session", None)
        if session is not None:
            self._session = None
            threading.Thread(
                target=session.shutdown,
                name="pdf-ocrer-shutdown",
                daemon=True,
            ).start()
        # 弹出菜单若开着，随窗口销毁即可；清掉引用防 dangling 调用。
        self._picker_menu = None
        self._close_picker_menu = None
        super().destroy()

    def _load_initial_gui_config(self) -> GuiConfig:
        try:
            cfg = load_config(self._config_path)
            setup_logging(cfg.logging)
            return cfg.gui
        except ConfigError as exc:
            setup_logging(LoggingConfig())
            _logger.error("设置错误: %s", exc)
            return GuiConfig()

    def _on_overwrite_toggled(self) -> None:
        """勾选变化立即写回配置，程序随时关闭都不丢勾选状态。"""
        try:
            set_overwrite_original(self._config_path, bool(self.overwrite_var.get()))
        except (ConfigError, OSError) as exc:
            _logger.warning("保存覆盖原文件偏好失败: %s", exc)

    def _load_initial_overwrite_pref(self) -> bool:
        try:
            return load_config(self._config_path).output.overwrite_original
        except ConfigError:
            return False

    def _build_widgets(self, initial_appearance: str) -> None:
        colors = ui_theme.palette(_effective_appearance(initial_appearance))
        self.configure(fg_color=colors["bg"])
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=3)
        self.rowconfigure(5, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 8))
        header.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="PDF OCR",
            font=ui_theme.font(22, "bold"),
            text_color=colors["text"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text=f"v{__version__}",
            font=ui_theme.font(14),
            text_color=colors["muted"],
        ).grid(row=0, column=1, sticky="e")

        input_card = ui_theme.card(self, colors)
        input_card.grid(row=1, column=0, sticky="ew", padx=20)
        input_card.columnconfigure(0, weight=1)

        self.paths_entry = ui_theme.field_entry(
            input_card,
            colors,
            textvariable=self.paths_var,
        )
        self.paths_entry.grid(row=0, column=0, sticky="ew", padx=(16, 8), pady=(12, 12))

        self.choose_button = ui_theme.primary_button(
            input_card,
            colors,
            "选择文件（夹）",
            self._show_picker_menu,
        )
        self.choose_button.grid(row=0, column=1, pady=(12, 12))
        # 悬停自动弹出（CTkButton.bind 会同时绑到 canvas 与文字标签，
        # 按钮任意区域都会触发）。点击仍可用，弹出有防重入保护。
        self.choose_button.bind("<Enter>", lambda _e: self._show_picker_menu())

        self.clear_button = ui_theme.secondary_button(
            input_card,
            colors,
            "清空",
            self._clear_paths,
            width=72,
        )
        self.clear_button.grid(row=0, column=2, padx=(8, 16), pady=(12, 12))

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 0))
        actions.columnconfigure(3, weight=1)

        self.start_button = ui_theme.primary_button(
            actions,
            colors,
            "开始处理",
            self._start,
        )
        self.start_button.grid(row=0, column=0)

        self.force_checkbox = ctk.CTkCheckBox(
            actions,
            text="重新处理",
            variable=self.force_var,
            font=ui_theme.font(14),
            text_color=colors["text"],
            fg_color=colors["accent"],
            hover_color=colors["accent_hover"],
            border_width=1,
            border_color=colors["field_border"],
            checkmark_color="#ffffff",
            checkbox_width=18,
            checkbox_height=18,
            corner_radius=5,
        )
        self.force_checkbox.grid(row=0, column=4, padx=(0, 12))

        self.cancel_button = ui_theme.secondary_button(
            actions,
            colors,
            "取消",
            self._cancel,
        )
        self.cancel_button.grid(row=0, column=1, padx=(16, 0))

        self.overwrite_checkbox = ctk.CTkCheckBox(
            actions,
            text="覆盖原文件",
            variable=self.overwrite_var,
            command=self._on_overwrite_toggled,
            font=ui_theme.font(14),
            text_color=colors["text"],
            fg_color=colors["accent"],
            hover_color=colors["accent_hover"],
            border_width=1,
            border_color=colors["field_border"],
            checkmark_color="#ffffff",
            checkbox_width=18,
            checkbox_height=18,
            corner_radius=5,
        )
        self.overwrite_checkbox.grid(row=0, column=5, padx=(0, 12))

        self.edit_config_button = ui_theme.secondary_button(
            actions,
            colors,
            "设置",
            self._open_settings_dialog,
            width=72,
        )
        self.edit_config_button.grid(row=0, column=6, sticky="e")

        progress_card = ui_theme.card(self, colors)
        progress_card.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 0))
        progress_card.columnconfigure(0, weight=1)

        self.progressbar = ctk.CTkProgressBar(
            progress_card,
            height=6,
            corner_radius=3,
            progress_color=colors["accent"],
            fg_color=colors["track"],
        )
        self.progressbar.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 6))
        self.progressbar.set(0)

        self.status_label = ctk.CTkLabel(
            progress_card,
            textvariable=self.status_var,
            anchor="w",
            font=ui_theme.font(14),
            text_color=colors["muted"],
        )
        self.status_label.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))

        table_card = ui_theme.card(self, colors)
        table_card.grid(row=4, column=0, sticky="nsew", padx=20, pady=(10, 0))
        table_card.columnconfigure(0, weight=1)
        table_card.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            table_card,
            text="文件列表",
            font=ui_theme.font(14, "bold"),
            text_color=colors["text"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))

        tree_holder = ctk.CTkFrame(table_card, fg_color="transparent")
        tree_holder.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        tree_holder.columnconfigure(0, weight=1)
        tree_holder.rowconfigure(0, weight=1)

        _style_treeview(initial_appearance, self)
        self.file_tree = ttk.Treeview(
            tree_holder,
            columns=_TREE_COLUMNS,
            show="headings",
            style="PdfOcrer.Treeview",
            height=6,
        )
        self.file_tree.heading("source", text="原文件名", anchor="w")
        self.file_tree.heading("status", text="状态", anchor="w")
        self.file_tree.heading("output", text="输出文件", anchor="w")
        self.file_tree.heading("ocr_pages", text="OCR页数", anchor="w")
        self.file_tree.column("source", width=280, anchor="w")
        self.file_tree.column("status", width=150, anchor="w")
        self.file_tree.column("output", width=280, anchor="w")
        self.file_tree.column("ocr_pages", width=90, anchor="center")
        self.file_tree.grid(row=0, column=0, sticky="nsew")
        self.file_tree.tag_configure("stripe", background=colors["heading"])

        tree_scrollbar = ctk.CTkScrollbar(
            tree_holder,
            orientation="vertical",
            command=self.file_tree.yview,
            width=10,
            corner_radius=5,
            fg_color="transparent",
            button_color=colors["secondary"],
            button_hover_color=colors["secondary_hover"],
        )
        tree_scrollbar.grid(row=0, column=1, sticky="ns", padx=(6, 0))

        # 超长单元格悬停提示：字体与行字体同像素，用于测量文字是否超宽
        scaling = float(ctk.ScalingTracker.get_widget_scaling(self))
        self._tree_hover_font = tkfont.Font(
            family=ui_theme.FONT_FAMILY, size=-round(14 * scaling)
        )
        self.file_tree.bind("<Motion>", self._on_tree_motion)
        self.file_tree.bind("<Leave>", lambda _event: self._hide_tree_tooltip())

        def _on_tree_yscroll(first: str, last: str) -> None:
            # 滚动列表时收起提示，避免它停留在与内容错位的位置
            self._hide_tree_tooltip()
            tree_scrollbar.set(first, last)

        self.file_tree.configure(yscrollcommand=_on_tree_yscroll)

        log_card = ui_theme.card(self, colors)
        log_card.grid(row=5, column=0, sticky="nsew", padx=20, pady=(10, 14))
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            log_card,
            text="运行日志",
            font=ui_theme.font(14, "bold"),
            text_color=colors["text"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))

        self.log_text = ctk.CTkTextbox(
            log_card,
            height=92,
            wrap="word",
            state="disabled",
            corner_radius=8,
            border_width=0,
            fg_color=colors["log_bg"],
            text_color=colors["text"],
            font=ui_theme.font(14),
            scrollbar_button_color=colors["secondary"],
            scrollbar_button_hover_color=colors["secondary_hover"],
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))

    def _setup_drag_drop(self) -> None:
        if not _DND_AVAILABLE or TkinterDnD is None or DND_FILES is None:
            return

        try:
            self.TkdndVersion = TkinterDnD._require(self)
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)
        except tk.TclError:  # tkdnd 扩展加载或注册失败：降级为不可拖放
            self._dnd_enabled = False
            return

        self._dnd_enabled = True

    def _on_drop(self, event: object) -> None:
        data = getattr(event, "data", "")
        self._add_input_paths(_paths_from_drop_data(data))

    def _add_input_paths(self, paths: Iterable[Path]) -> None:
        existing = {str(path).casefold() for path in self._input_paths}
        added_count = 0
        rejected = 0
        outputs = 0
        for path in paths:
            # 本工具的输出文件（*_OCR.pdf）绝不作为输入：既无意义，覆盖模式
            # 下还会把上次的输出改写掉。文件夹交给管线扫描时同样会排除。
            if path.is_file() and is_generated_output(path):
                outputs += 1
                continue
            # 文件按扩展名过滤，txt 等直接忽略
            if (
                path.is_file()
                and path.suffix.removeprefix(".").casefold() not in _DROP_ALLOWED_EXTENSIONS
            ):
                rejected += 1
                continue
            key = str(path).casefold()
            if key not in existing:
                self._input_paths.append(path)
                existing.add(key)
                added_count += 1
        notes = []
        if added_count:
            notes.append(f"已添加 {added_count} 个文件（夹）")
        if outputs:
            notes.append(f"已忽略 {outputs} 个输出文件（*_OCR.pdf 不需要再次识别）")
        if rejected:
            notes.append(f"已忽略 {rejected} 个不支持的文件（仅支持 PDF 和图片）")
        if notes:
            self.status_var.set("；".join(notes))
        if added_count:
            self._sync_paths_entry()

    def _clear_paths(self) -> None:
        self._input_paths.clear()
        self._sync_paths_entry()

    def _sync_paths_entry(self) -> None:
        self.paths_var.set("; ".join(str(path) for path in self._input_paths))
        # 内容超长时从末尾展示：新拖入/选择的路径立即可见
        self.paths_entry.xview_moveto(1.0)

    def _build_picker_menu(self) -> tk.Toplevel:
        """构建「选择文件/文件夹」弹出菜单（自绘条目）。

        原生 tk.Menu 在 Windows 上的行高由系统固定绘制、无法配置，高分
        辨率下菜单项拥挤；改用 tk.Label 自绘：字体用负像素随 CTk 缩放，
        行高 = 字体 + 上下留白，同样按缩放折算。
        """
        scaling = ctk.ScalingTracker.get_widget_scaling(self)
        colors = ui_theme.palette(ui_theme.current_appearance())
        item_font = (ui_theme.FONT_FAMILY, -round(14 * scaling))
        pad_y = max(6, round(9 * scaling))
        pad_x = max(10, round(14 * scaling))
        hover_bg = colors["secondary"]
        hover_fg = colors["select_text"]

        popup = tk.Toplevel(self)
        popup.overrideredirect(True)
        popup.withdraw()  # 定位完成前先隐藏，避免在屏幕原点闪现
        popup.configure(bg=colors["card"])

        def close(command=None) -> None:
            if getattr(self, "_picker_menu", None) is not popup:
                return
            self._picker_menu = None
            try:
                popup.grab_release()
            except tk.TclError:
                pass
            try:
                popup.destroy()
            except tk.TclError:
                pass
            if command is not None:
                # 选中条目后指针常留在按钮上：短暂抑制 hover 自动弹出，
                # 防止文件对话框弹出前旧位置又触发一个新菜单。
                self._picker_suspend_until = time.monotonic() + 1.2
                # 等 grab 完全释放后再打开文件对话框。
                self.after(50, command)

        for label, command in (
            (_PICK_FILE_LABEL, self._choose_files),
            (_PICK_FOLDER_LABEL, self._choose_folder),
        ):
            item = tk.Label(
                popup,
                text=label,
                font=item_font,
                anchor="w",
                bg=colors["card"],
                fg=colors["text"],
                padx=pad_x,
                pady=pad_y,
            )
            item.pack(fill="x")
            item.bind(
                "<Enter>",
                lambda _e, w=item: w.configure(bg=hover_bg, fg=hover_fg),
            )
            item.bind(
                "<Leave>",
                lambda _e, w=item: w.configure(bg=colors["card"], fg=colors["text"]),
            )
            item.bind("<Button-1>", lambda _e, c=command: close(c))

        self._picker_menu = popup
        self._close_picker_menu = close
        return popup

    def _show_picker_menu(self) -> None:
        menu = getattr(self, "_picker_menu", None)
        if menu is not None:
            # 残留引用检查：菜单窗口被外部销毁（如主窗口销毁连带）后必须
            # 清掉引用，否则悬停再也弹不出菜单。
            try:
                if menu.winfo_exists():
                    return
            except tk.TclError:
                pass
            self._picker_menu = None
        if time.monotonic() < getattr(self, "_picker_suspend_until", 0.0):
            return

        popup = self._build_picker_menu()
        button_path = str(self.choose_button)

        def watch_pointer() -> None:
            """无 grab 的菜单靠周期自检关闭：指针离开菜单和按钮即收起。

            不依赖键盘焦点（overrideredirect 窗口在 Windows 上通常拿不到
            焦点，FocusOut/Esc 并不总是可靠）。
            """
            if self._picker_menu is not popup:
                return
            under = popup.winfo_containing(popup.winfo_pointerx(), popup.winfo_pointery())
            under_path = str(under) if under is not None else ""
            on_menu = under_path.startswith(str(popup))
            on_button = under_path == button_path or under_path.startswith(button_path + ".")
            if under_path and (on_menu or on_button):
                popup.after(250, watch_pointer)
                return
            self._close_picker_menu()

        popup.geometry(
            f"+{self.choose_button.winfo_rootx()}+"
            f"{self.choose_button.winfo_rooty() + self.choose_button.winfo_height()}"
        )
        popup.deiconify()
        popup.lift()
        popup.focus_set()
        # 不使用 grab_set：全局事件抓取会吞掉主窗口的一切操作（关闭界面、
        # 开始处理等全部失效）。关闭靠指针自检（上方）与 FocusOut 兜底，
        # 点击主界面其它位置时菜单关闭且该次点击本身正常生效。
        popup.bind("<FocusOut>", lambda _e: self._close_picker_menu())
        popup.bind("<Escape>", lambda _e: self._close_picker_menu())
        popup.after(250, watch_pointer)

    def _choose_files(self) -> None:
        selected = filedialog.askopenfilenames(
            parent=self,
            title="选择 PDF 文件",
            filetypes=[
                ("PDF 和图片", "*.pdf *.jpg *.jpeg *.png *.tif *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        if selected:
            self._add_input_paths(Path(path) for path in selected)

    def _choose_folder(self) -> None:
        selected = filedialog.askdirectory(parent=self, title="选择文件夹")
        if selected:
            self._add_input_paths((Path(selected),))

    def _parse_paths_entry(self) -> list[Path]:
        text = self.paths_var.get().strip().strip(";").strip()
        if not text:
            return []
        targets: list[Path] = []
        seen: set[str] = set()
        for part in text.split(";"):
            part = part.strip()
            if not part:
                continue
            key = part.casefold()
            if key in seen:
                continue
            seen.add(key)
            targets.append(Path(part))
        return targets

    def _start(self) -> None:
        targets = self._parse_paths_entry()
        if not targets:
            messagebox.showerror("路径错误", "请先选择要处理的文件或文件夹", parent=self)
            return

        # 不在这里拦截不存在的路径：交给批处理按条目失败，
        # 文件列表会给出明确的「失败」行与原因，不弹窗阻断。

        self._clear_log()
        self._clear_file_rows()
        self.progressbar.set(0)
        self._run_started_at = time.monotonic()
        self._progress_seen = False
        self._page_started_at.clear()
        self._page_durations.clear()
        self._file_started_at.clear()
        self.status_var.set("正在后台加载 OCR 模型…（完成后自动开始处理）")
        self._cancel_event = threading.Event()
        # 防重入：防御分支提前解锁 UI 时，不允许并发启动第二个批次。
        if self._worker is not None and self._worker.is_alive():
            return
        force = bool(self.force_var.get())
        overwrite = bool(self.overwrite_var.get())
        self._set_running(True)
        self._cancel_status_timer()
        self._status_after_id = self.after(1000, self._tick_status)

        self._worker = threading.Thread(
            target=self._run_worker,
            args=(targets, self._cancel_event, force, overwrite),
            daemon=False,
        )
        self._worker.start()

    def _cancel(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self._append_log("正在取消处理...")

    def _run_worker(
        self,
        targets: list[Path],
        cancel_event: threading.Event,
        force: bool | None = None,
        overwrite_original: bool = False,
    ) -> None:
        try:
            cfg = load_config(self._config_path)
            cfg = replace(cfg, output=replace(cfg.output, overwrite_original=overwrite_original))
            setup_logging(cfg.logging)
            log_cb = self._queue_log
            session = self._session
            if session is None:
                # 窗口已销毁（destroy 把 session 置 None）时不再新建：
                # 新建的 session 将永远不会被 shutdown，子进程会变孤儿。
                if self._closed:
                    return
                session = WorkerSession()
                self._session = session
            if self._engine_factory is not None:
                # 注入引擎的调用方（单元测试）：在线程内顺序执行。
                engine = self._create_engine(cfg.ocr, log_cb)
                summary = run_batch(
                    targets,
                    cfg,
                    engine,
                    progress_cb=self._queue_progress,
                    log_cb=log_cb,
                    cancel_event=cancel_event,
                    file_cb=self._queue_file_done,
                    pending_cb=self._queue_files,
                    page_start_cb=self._queue_page_start,
                    force=bool(force),
                )
            else:
                # 生产路径：OCR 在常驻子进程中顺序执行（一次一个文件）。
                # ONNX 推理是 C++ 密集操作且不释放 GIL，若在本进程线程内
                # 执行会把 Tk 事件循环饿死数十秒（窗口显示"无响应"）；
                # 子进程执行后 UI 进程完全不加载引擎，且模型只在首次批次
                # 加载，之后复用到程序关闭。
                summary = session.run_batch(
                    targets,
                    cfg,
                    None,
                    progress_cb=self._queue_progress,
                    log_cb=log_cb,
                    cancel_event=cancel_event,
                    file_cb=self._queue_file_done,
                    pending_cb=self._queue_files,
                    page_start_cb=self._queue_page_start,
                    force=bool(force),
                )
        except Exception as exc:
            _logger.exception("GUI worker failed")
            self._queue.put(("error", f"{type(exc).__name__}: {exc}"))
        else:
            self._queue.put(("done", summary))

    def _create_engine(
        self,
        cfg: OcrConfig,
        log_cb: Callable[[str], None],
    ) -> OcrEngineProtocol:
        if self._engine_factory is not None:
            return self._engine_factory(cfg)

        return create_engine(cfg, log_cb)

    def _queue_log(self, message: str) -> None:
        self._queue.put(("log", message))

    def _queue_progress(
        self,
        file_i: int,
        file_n: int,
        page_i: int,
        page_n: int,
        name: str,
    ) -> None:
        self._queue.put(("progress", file_i, file_n, page_i, page_n, name))

    def _queue_file_done(self, result: FileResult) -> None:
        self._queue.put(("file_done", result))

    def _queue_files(self, rels: list[str]) -> None:
        self._queue.put(("files", rels))

    def _queue_page_start(
        self,
        file_i: int,
        file_n: int,
        page_i: int,
        page_n: int,
        name: str,
    ) -> None:
        self._queue.put(("page_start", file_i, file_n, page_i, page_n, name))

    def _poll_queue(self) -> None:
        self._drain_queue()
        if not self._closed:
            self._after_id = self.after(100, self._poll_queue)

    def _drain_queue(self) -> None:
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break

            if not event:
                continue

            kind = event[0]
            if kind == "log":
                self._append_log(str(event[1]))
            elif kind == "files":
                self._prefill_rows(event[1])
            elif kind == "page_start":
                self._handle_page_start(event)
            elif kind == "progress":
                self._handle_progress(event)
            elif kind == "file_done":
                self._handle_file_done(event[1])
            elif kind == "done":
                self._handle_done(event[1])
            elif kind == "error":
                self._handle_error(str(event[1]))
            else:
                # 未知事件不阻断界面，记录到文件日志便于排查。
                _logger.warning("GUI 收到未知事件类型: %s", kind)

    def _handle_page_start(self, event: GuiEvent) -> None:
        """页开始：模型加载完成后立即给出"识别中"反馈。"""
        self._update_page_state(event)
        if self._current_page_i == 1 or self._current_file_i not in self._file_started_at:
            self._file_started_at[self._current_file_i] = time.monotonic()
        self._page_started_at[self._current_file_i] = time.monotonic()
        if self._current_page_i == 1:
            self._page_durations[self._current_file_i] = []
        if self._row_is_unfinished(self._current_name):
            display = _display_rel(self._current_name)
            self._upsert_file_row(self._current_name, "处理中", "", "", display_name=display)
        self.progressbar.set(self._fraction(self._current_page_i - 1))
        self.status_var.set(self._status_text())

    def _handle_progress(self, event: GuiEvent) -> None:
        """页完成：记录本页耗时并推进进度。"""
        self._update_page_state(event)
        started = self._page_started_at.pop(self._current_file_i, None)
        if started is not None:
            self._page_durations.setdefault(self._current_file_i, []).append(
                max(0.0, time.monotonic() - started)
            )
        if self._row_is_unfinished(self._current_name):
            display = _display_rel(self._current_name)
            self._upsert_file_row(self._current_name, "处理中", "", "", display_name=display)
        self.progressbar.set(self._fraction(self._current_page_i))
        self.status_var.set(self._status_text())

    def _update_page_state(self, event: GuiEvent) -> None:
        file_i, file_n, page_i, page_n, name = event[1:6]
        self._progress_seen = True
        self._current_file_i = int(file_i)
        self._current_file_n = int(file_n)
        self._current_page_i = int(page_i)
        self._current_page_n = int(page_n)
        self._current_name = str(name)

    def _fraction(self, pages_done: float) -> float:
        if self._current_file_n < 1 or self._current_page_n < 1:
            return 0.0
        value = (
            self._current_file_i - 1 + pages_done / self._current_page_n
        ) / self._current_file_n
        return max(0.0, min(1.0, value))

    def _tick_status(self) -> None:
        self._status_after_id = None
        if not self._running or self._closed:
            return
        self.status_var.set(self._status_text())
        self._status_after_id = self.after(1000, self._tick_status)

    def _status_text(self) -> str:
        if not self._progress_seen:
            return (
                f"正在后台加载 OCR 模型…（已 {self._elapsed_seconds()} 秒，"
                "完成后自动开始处理）"
            )

        text = (
            f"第 {self._current_file_i}/{self._current_file_n} 个 - "
            f"第 {self._current_page_i}/{self._current_page_n} 页 - "
            f"{_display_rel(self._current_name)}"
        )
        started = self._file_started_at.get(self._current_file_i)
        if started is None:
            return text

        elapsed = int(time.monotonic() - started)
        text += f"（识别中… 本文件已用 {elapsed} 秒"
        eta = self._page_eta_seconds()
        if eta is not None:
            text += f"，预计还需 ~{eta} 秒"
        return text + "）"

    def _page_eta_seconds(self) -> int | None:
        remaining = self._current_page_n - self._current_page_i
        durations = self._page_durations.get(self._current_file_i, [])
        if remaining <= 0 or not durations:
            return None
        average = sum(durations) / len(durations)
        return max(1, round(average * remaining))

    def _elapsed_seconds(self) -> int:
        if self._run_started_at is None:
            return 0
        return int(max(0.0, time.monotonic() - self._run_started_at))

    def _cancel_status_timer(self) -> None:
        if self._status_after_id is None:
            return
        try:
            self.after_cancel(self._status_after_id)
        except tk.TclError:
            pass
        self._status_after_id = None

    def _prefill_rows(self, rels: object) -> None:
        """开始处理时先列出全部待处理文件，避免"处理一个才出现一个"。"""
        if not isinstance(rels, list | tuple):
            return
        for rel in rels:
            rel = str(rel)
            self._upsert_file_row(rel, "等待中", "", "", display_name=_display_rel(rel))

    def _handle_file_done(self, result: object) -> None:
        if not isinstance(result, FileResult):
            self._handle_error("GUI 收到未知的文件完成事件。")
            return

        rel = _result_source_name(result)
        self._upsert_file_row(
            rel,
            result.status.value,
            _result_output_name(result),
            str(result.ocr_pages),
            display_name=_display_rel(rel),
        )
        # 本文件结束：清掉它的计时，状态栏不再显示已用时间。
        self._file_started_at.pop(self._current_file_i, None)
        self._page_started_at.pop(self._current_file_i, None)
        self._page_durations.pop(self._current_file_i, None)
        self.status_var.set(self._status_text())

    def _handle_done(self, summary: object) -> None:
        if not isinstance(summary, BatchSummary):
            self._handle_error("GUI 收到未知的完成事件。")
            return

        was_running = self._running
        self._cancel_status_timer()
        self._set_running(False)
        self._file_started_at.clear()
        if not summary.cancelled:
            self.progressbar.set(1.0 if summary.results else 0.0)
        self.status_var.set(
            "已取消" if summary.cancelled else f"完成：共 {len(summary.results)} 个文件"
        )
        if summary.cancelled:
            self._mark_unfinished_rows_cancelled()
        if was_running:
            self._append_log(self._completion_message(summary))

    def _mark_unfinished_rows_cancelled(self) -> None:
        for iid in self._file_rows.values():
            values = self.file_tree.item(iid, "values")
            if len(values) < 2 or values[1] not in {"等待中", "处理中"}:
                continue
            # 保留首列的显示名（含消歧前缀的 Windows 分隔符），不要回退成内部键。
            self.file_tree.item(iid, values=(values[0], "已取消", "", ""))

    def _handle_error(self, message: str) -> None:
        was_running = self._running
        self._cancel_status_timer()
        self._set_running(False)
        self.status_var.set("发生错误")
        self._append_log(message)
        if was_running:
            self.after(0, lambda: messagebox.showerror("错误", message, parent=self))

    def _set_running(self, running: bool) -> None:
        self._running = running
        if running:
            self.start_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")
        else:
            self.start_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _on_tree_motion(self, event: tk.Event) -> None:
        """鼠标在文件列表上移动：仅当单元格文字超宽时显示完整内容提示。"""
        if self._tree_hover_font is None:
            return
        row_id = self.file_tree.identify_row(event.y)
        col_id = self.file_tree.identify_column(event.x)
        if not row_id or not col_id.startswith("#"):
            self._hide_tree_tooltip()
            return
        col_index = int(col_id[1:]) - 1
        values = self.file_tree.item(row_id, "values")
        if col_index < 0 or col_index >= len(values):
            self._hide_tree_tooltip()
            return
        text = str(values[col_index])
        col_width = int(self.file_tree.column(col_id, "width"))
        if not text or self._tree_hover_font.measure(text) <= col_width - 16:
            self._hide_tree_tooltip()
            return
        key = (row_id, col_id, text)
        if self._tree_tooltip_key == key and self._tree_tooltip is not None:
            return  # 同一单元格已在提示，避免闪烁
        self._tree_tooltip_key = key
        self._show_tree_tooltip(text, event.x_root, event.y_root)

    def _show_tree_tooltip(self, text: str, x_root: int, y_root: int) -> None:
        colors = ui_theme.palette(_effective_appearance(str(ctk.get_appearance_mode())))
        if self._tree_tooltip is None:
            self._tree_tooltip = tk.Toplevel(self)
            self._tree_tooltip.withdraw()
            self._tree_tooltip.overrideredirect(True)
            self._tree_tooltip_label = tk.Label(self._tree_tooltip, justify="left")
            self._tree_tooltip_label.pack()
        scaling = float(ctk.ScalingTracker.get_widget_scaling(self))
        label = self._tree_tooltip_label
        assert label is not None
        label.configure(
            text=text,
            font=(ui_theme.FONT_FAMILY, -round(13 * scaling)),
            wraplength=int(460 * scaling),
            bg=colors["card"],
            fg=colors["text"],
            highlightthickness=1,
            highlightbackground=colors["card_border"],
            padx=8,
            pady=5,
        )
        self._tree_tooltip.geometry(f"+{x_root + 14}+{y_root + 18}")
        self._tree_tooltip.deiconify()
        self._tree_tooltip.lift()

    def _hide_tree_tooltip(self) -> None:
        self._tree_tooltip_key = None
        if self._tree_tooltip is not None:
            self._tree_tooltip.withdraw()

    def _clear_file_rows(self) -> None:
        self._hide_tree_tooltip()
        for iid in self.file_tree.get_children():
            self.file_tree.delete(iid)
        self._file_rows.clear()

    def _row_is_unfinished(self, source_name: str) -> bool:
        """行不存在或仍处等待/处理中时才允许进度事件刷新它。

        防御跨进程事件乱序：迟到的页事件不得把已完成的行刷回"处理中"。
        """
        iid = self._file_rows.get(source_name)
        if iid is None:
            return True
        values = self.file_tree.item(iid, "values")
        return len(values) >= 2 and values[1] in {"等待中", "处理中"}

    def _upsert_file_row(
        self,
        source_name: str,
        status: str,
        output_name: str,
        ocr_pages: str,
        display_name: str | None = None,
    ) -> None:
        """按 rel（唯一键）更新行；首列显示纯文件名，消歧前缀不外显。"""
        iid = self._file_rows.get(source_name)
        values = (display_name or source_name, status, output_name, ocr_pages)
        if iid is None:
            iid = f"file-{len(self._file_rows)}"
            tags = ("stripe",) if len(self._file_rows) % 2 == 1 else ()
            self._file_rows[source_name] = iid
            self.file_tree.insert("", "end", iid=iid, values=values, tags=tags)
            return

        self.file_tree.item(iid, values=values)

    def _completion_message(self, summary: BatchSummary) -> str:
        failed = sum(result.status is FileStatus.FAILED for result in summary.results)
        skipped = sum(result.status is FileStatus.SKIPPED_ENCRYPTED for result in summary.results)
        skipped_done = sum(result.status is FileStatus.SKIPPED_DONE for result in summary.results)
        existing = sum(
            result.status is FileStatus.SUCCESS_EXISTING_TEXT for result in summary.results
        )
        no_text = sum(result.status is FileStatus.NO_TEXT_FOUND for result in summary.results)
        # 成功口径 = 仅 OCR 完成（产出了新的可搜索 PDF）；其余全部归入跳过。
        success = len(summary.results) - failed - skipped - skipped_done - existing - no_text
        state = "已取消" if summary.cancelled else "完成"
        return (
            f"状态：{state}｜成功：{success}｜"
            f"跳过：{skipped + skipped_done + existing + no_text}"
            f"（加密 {skipped}、已输出 {skipped_done}、"
            f"已OCR {existing}、无文本 {no_text}）｜"
            f"失败：{failed}"
        )

    def _open_settings_dialog(self) -> None:
        try:
            dialog = SettingsDialog(self, self._config_path)
        except ConfigError as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return
        dialog.grab_set()
        self.wait_window(dialog)

    def _on_close(self) -> None:
        if self._running:
            confirmed = messagebox.askokcancel(
                "取消处理",
                "处理仍在进行，要取消并关闭吗？",
                parent=self,
            )
            if not confirmed:
                return
            if self._cancel_event is not None:
                self._cancel_event.set()
        self.destroy()


def _result_source_name(result: FileResult) -> str:
    return result.rel or result.source.name


def _display_rel(rel: str) -> str:
    """表格/状态栏的显示名：保留同名消歧前缀，分隔符用 Windows 的 \\。

    内部行键（rel）统一为 / 分隔（与扫描逻辑一致）；仅在显示时转换为
    Windows 风格，方便用户按文件夹区分同名文件。
    """
    return rel.replace("/", "\\")


def _result_output_name(result: FileResult) -> str:
    if result.output is None:
        return ""
    return result.output.name
