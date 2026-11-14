"""Treeview 字体与 CTk 字体缩放一致性回归测试。

文件列表的 ttk.Treeview 是界面唯一的 ttk 控件：ttk 字体的「点」按
dpi/72 折算，CTk 字体按 dpi/96 折算，同样字号差 4/3。因此
_style_treeview 用固定像素（负值）并按 CTk 同源的 widget scaling 换算。
本测试端到端保证该公式与 CTk 控件渲染 CTkFont 的公式在常见缩放档位下
逐档相等（各读各的底层真实渲染值，不复述公式）。
"""
from __future__ import annotations

import re

import pytest

tk = pytest.importorskip("tkinter")
ctk = pytest.importorskip("customtkinter")
ttk = pytest.importorskip("tkinter.ttk")

from customtkinter import ScalingTracker

from pdf_ocrer import ui_theme
from pdf_ocrer.gui import _style_treeview

_SCALINGS = (1.0, 1.25, 1.5)


def _font_px(raw: object) -> int:
    """从 Tcl 字体描述中提取负像素字号（串中唯一负数）。"""
    return -abs(int(re.search(r"(-\d+)", str(raw)).group(1)))


def test_treeview_font_matches_ctk_rendering_across_scalings() -> None:
    try:
        root = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()

    style = ttk.Style()
    try:
        for scaling in _SCALINGS:
            ScalingTracker.set_widget_scaling(scaling)
            root.update_idletasks()

            # CTk 侧：控件渲染 CTkFont(14) 后底层 tk.Label 的真实字号
            label = ctk.CTkLabel(root, text="测", font=ui_theme.font(14))
            label.grid()
            root.update_idletasks()
            try:
                # _label 是 CTkLabel 内部真正的文字控件；canvas 不承载文字
                ctk_px = _font_px(label._label.cget("font"))
            finally:
                label.destroy()

            # Treeview 侧：_style_treeview 配置后的真实字号
            _style_treeview("light", root)
            tree_px = _font_px(style.lookup("PdfOcrer.Treeview", "font"))

            assert tree_px == ctk_px, (
                f"scaling {scaling}: Treeview 字号 {tree_px}px != CTk 渲染 {ctk_px}px"
            )
    finally:
        # ScalingTracker 是进程级全局状态，必须还原，避免影响其它 GUI 测试
        ScalingTracker.set_widget_scaling(1.0)
        root.destroy()
