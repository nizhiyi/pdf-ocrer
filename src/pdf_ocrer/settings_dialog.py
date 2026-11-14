from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from pdf_ocrer import ui_theme
from pdf_ocrer.config import (
    CommonSettings,
    ConfigError,
    apply_common_settings,
    read_common_settings,
)

# 与 RapidOCR default_models.yaml 一致：v6 只有 tiny/small/medium，
# v4/v5 只有 mobile/server。
OCR_VERSIONS = ("PP-OCRv4", "PP-OCRv5", "PP-OCRv6")

MODEL_TYPES_BY_VERSION: dict[str, tuple[str, ...]] = {
    "PP-OCRv4": ("mobile", "server"),
    "PP-OCRv5": ("mobile", "server"),
    "PP-OCRv6": ("tiny", "small", "medium"),
}

DEFAULT_MODEL_TYPE_BY_VERSION: dict[str, str] = {
    "PP-OCRv4": "mobile",
    "PP-OCRv5": "mobile",
    "PP-OCRv6": "small",
}

# rapidocr execution device.
DEVICE_VALUES = ("cpu", "cuda", "dml")


def model_types_for_version(ocr_version: str) -> tuple[str, ...]:
    return MODEL_TYPES_BY_VERSION.get(ocr_version, MODEL_TYPES_BY_VERSION["PP-OCRv6"])


def resolve_model_type(ocr_version: str, model_type: str) -> str:
    """模型大小不在该版本的合法集合时，回退到该版本的默认值。"""
    allowed = model_types_for_version(ocr_version)
    return model_type if model_type in allowed else DEFAULT_MODEL_TYPE_BY_VERSION.get(
        ocr_version, "small"
    )


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, master: object, config_path: Path) -> None:
        super().__init__(master)
        self._config_path = Path(config_path)
        self._colors = ui_theme.palette(ui_theme.current_appearance())

        settings = read_common_settings(self._config_path)
        # 旧配置里的版本↔大小组合可能非法（如 v6 + server），加载时归一。
        self._ocr_version = settings.ocr_version
        self._model_type = resolve_model_type(settings.ocr_version, settings.model_type)

        self.title("设置")
        self.resizable(False, False)
        self.configure(fg_color=self._colors["bg"])
        self.columnconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._build_header()
        self._build_ocr_section(settings)
        self._build_buttons()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 10))
        header.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="设置",
            font=ui_theme.font(22, "bold"),
            text_color=self._colors["text"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="调整 OCR 模型；保存后写入 config.toml",
            font=ui_theme.font(14),
            text_color=self._colors["muted"],
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

    def _build_ocr_section(self, settings: CommonSettings) -> None:
        frame = ui_theme.card(self, self._colors)
        frame.grid(row=1, column=0, sticky="ew", padx=20)
        frame.columnconfigure(1, weight=1)

        self._section_title(frame, "OCR 设置", row=0)

        self._label(frame, "模型版本", row=0)
        self.ocr_version_var = tk.StringVar(master=self, value=self._ocr_version)
        self.ocr_version_menu = self._option_menu(
            frame, list(OCR_VERSIONS), self.ocr_version_var, row=1
        )
        self.ocr_version_menu.set(self._ocr_version)
        self.ocr_version_menu.configure(command=self._on_version_changed)
        self.ocr_version_help_label = self._help_label(
            frame, "安装包仅内置 PP-OCRv6 small；其它版本/大小首次使用需联网下载"
        )
        self.ocr_version_help_label.grid(row=2, column=1, sticky="w", padx=(8, 16))

        self._label(frame, "模型大小", row=2)
        self.model_type_var = tk.StringVar(master=self, value=self._model_type)
        self.model_type_menu = self._option_menu(
            frame,
            list(model_types_for_version(self._ocr_version)),
            self.model_type_var,
            row=3,
        )
        self.model_type_menu.set(self._model_type)

        self._label(frame, "执行设备", row=3)
        self.device_var = tk.StringVar(master=self, value=settings.device)
        self.device_menu = self._option_menu(
            frame, list(DEVICE_VALUES), self.device_var, row=4
        )
        self.device_menu.set(settings.device)
        self.device_help_label = self._help_label(
            frame, "cuda/dml 需另装对应 GPU 版 onnxruntime 套件"
        )
        self.device_help_label.grid(row=5, column=1, sticky="w", padx=(8, 16))

        self._label(frame, "DPI", row=5)
        self.dpi_entry = ui_theme.field_entry(frame, self._colors)
        self.dpi_entry.grid(row=6, column=1, sticky="ew", padx=(8, 16), pady=6)
        self._set_entry_text(self.dpi_entry, str(settings.dpi))

        self._label(frame, "最低信心分数", row=6)
        self.min_confidence_entry = ui_theme.field_entry(frame, self._colors)
        self.min_confidence_entry.grid(row=7, column=1, sticky="ew", padx=(8, 16), pady=6)
        self._set_entry_text(self.min_confidence_entry, str(settings.min_confidence))
        self.min_confidence_help_label = self._help_label(frame, "低于该分数的文字行会被丢弃")
        self.min_confidence_help_label.grid(row=8, column=1, sticky="w", padx=(8, 16), pady=(0, 14))

    def _on_version_changed(self, version: str) -> None:
        """版本切换：模型大小选项联动刷新，非法值回退到该版本默认。"""
        allowed = list(model_types_for_version(version))
        self.model_type_menu.configure(values=allowed)
        current = self.model_type_var.get()
        if current not in allowed:
            fallback = resolve_model_type(version, current)
            self.model_type_var.set(fallback)
            self.model_type_menu.set(fallback)

    def _build_buttons(self) -> None:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(14, 18))
        frame.columnconfigure(0, weight=1)

        self.save_button = ui_theme.primary_button(frame, self._colors, "保存", self._on_save)
        self.save_button.grid(row=0, column=1, padx=(8, 0))

        self.cancel_button = ui_theme.secondary_button(
            frame, self._colors, "取消", self._on_cancel, width=88
        )
        self.cancel_button.grid(row=0, column=2, padx=(8, 0))

    def _section_title(self, master: ctk.CTkFrame, text: str, row: int) -> None:
        ctk.CTkLabel(
            master,
            text=text,
            font=ui_theme.font(14, "bold"),
            text_color=self._colors["text"],
            anchor="w",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 8))

    def _label(self, master: ctk.CTkFrame, text: str, row: int) -> None:
        ctk.CTkLabel(
            master,
            text=text,
            font=ui_theme.font(14),
            text_color=self._colors["text"],
            anchor="w",
        ).grid(row=row + 1, column=0, sticky="w", padx=(16, 0), pady=6)

    def _help_label(self, master: ctk.CTkFrame, text: str) -> ctk.CTkLabel:
        return ctk.CTkLabel(
            master,
            text=text,
            font=ui_theme.font(12),
            text_color=self._colors["muted"],
            anchor="w",
        )

    def _option_menu(
        self,
        master: ctk.CTkFrame,
        values: list[str],
        variable: tk.StringVar,
        *,
        row: int,
    ) -> ctk.CTkOptionMenu:
        holder = ctk.CTkFrame(
            master,
            fg_color=self._colors["field"],
            corner_radius=8,
            border_width=1,
            border_color=self._colors["field_border"],
            height=34,
        )
        holder.grid(row=row, column=1, sticky="ew", padx=(8, 16), pady=6)
        holder.grid_propagate(False)

        menu = ctk.CTkOptionMenu(
            holder,
            values=values,
            variable=variable,
            height=30,
            corner_radius=6,
            fg_color=self._colors["field"],
            button_color=self._colors["field"],
            button_hover_color=self._colors["secondary"],
            text_color=self._colors["text"],
            font=ui_theme.font(14),
            dropdown_fg_color=self._colors["card"],
            dropdown_hover_color=self._colors["secondary"],
            dropdown_text_color=self._colors["text"],
            dropdown_font=ui_theme.font(14),
            anchor="w",
            dynamic_resizing=False,
        )
        menu.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(0, weight=1)
        return menu

    def _set_entry_text(self, entry: ctk.CTkEntry, text: str) -> None:
        entry.delete(0, "end")
        entry.insert(0, text)

    def _on_save(self) -> None:
        try:
            settings = CommonSettings(
                ocr_version=self.ocr_version_var.get(),
                dpi=int(self.dpi_entry.get().strip()),
                min_confidence=float(self.min_confidence_entry.get().strip()),
                device=self.device_var.get(),
                model_type=self.model_type_var.get(),
            )
            apply_common_settings(self._config_path, settings)
        except (ConfigError, ValueError) as exc:
            messagebox.showerror("设置错误", str(exc), parent=self)
            return
        except OSError as exc:
            messagebox.showerror("设置错误", f"无法写入设置文件：{exc}", parent=self)
            return

        self.destroy()

    def _on_cancel(self) -> None:
        self.destroy()
