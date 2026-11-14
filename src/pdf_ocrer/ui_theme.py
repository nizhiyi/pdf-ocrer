"""Shared visual language for the GUI and the settings dialog.

Flat, card-based layout with a single accent colour, similar to modern app
shells: neutral surfaces, subtle borders, rounded corners and a clear type
hierarchy. Both light and dark palettes follow the effective appearance mode.
"""

from __future__ import annotations

from typing import Final

import customtkinter as ctk

FONT_FAMILY: Final = "Microsoft YaHei UI"

LightPalette = dict[str, str]

_LIGHT: Final[LightPalette] = {
    "bg": "#f5f6f8",
    "card": "#ffffff",
    "card_border": "#e5e7eb",
    "text": "#111827",
    "muted": "#6b7280",
    "accent": "#2563eb",
    "accent_hover": "#1d4ed8",
    "secondary": "#f3f4f6",
    "secondary_hover": "#e5e7eb",
    "field": "#f9fafb",
    "field_border": "#d7dbe3",
    "heading": "#f3f4f6",
    "tree_bg": "#ffffff",
    "select": "#dbeafe",
    "select_text": "#111827",
    "log_bg": "#f8fafc",
    "track": "#e5e7eb",
}

_DARK: Final[LightPalette] = {
    "bg": "#17181c",
    "card": "#1f2127",
    "card_border": "#2b2e36",
    "text": "#e6e7eb",
    "muted": "#9aa0ab",
    "accent": "#4c8dff",
    "accent_hover": "#3b78e0",
    "secondary": "#2a2d35",
    "secondary_hover": "#333742",
    "field": "#23262d",
    "field_border": "#343841",
    "heading": "#242730",
    "tree_bg": "#1f2127",
    "select": "#2f4f8f",
    "select_text": "#ffffff",
    "log_bg": "#1b1d22",
    "track": "#2a2d35",
}


def current_appearance() -> str:
    """Resolve the active customtkinter appearance to ``light`` or ``dark``."""
    getter = getattr(ctk, "get_appearance_mode", None)
    mode = str(getter()).casefold() if callable(getter) else "system"
    return "dark" if mode == "dark" else "light"


def palette(appearance: str) -> LightPalette:
    return _DARK if appearance.casefold() == "dark" else _LIGHT


def font(size: int, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=FONT_FAMILY, size=size, weight=weight)


def card(master: object, colors: LightPalette) -> ctk.CTkFrame:
    """Rounded surface used for every section of the main window."""
    return ctk.CTkFrame(
        master,  # type: ignore[arg-type]
        fg_color=colors["card"],
        corner_radius=12,
        border_width=1,
        border_color=colors["card_border"],
    )


def primary_button(master: object, colors: LightPalette, text: str, command: object) -> ctk.CTkButton:
    return ctk.CTkButton(
        master,  # type: ignore[arg-type]
        text=text,
        command=command,  # type: ignore[arg-type]
        height=38,
        corner_radius=8,
        fg_color=colors["accent"],
        hover_color=colors["accent_hover"],
        text_color="#ffffff",
        font=font(14),
    )


def secondary_button(
    master: object,
    colors: LightPalette,
    text: str,
    command: object,
    *,
    width: int = 96,
) -> ctk.CTkButton:
    return ctk.CTkButton(
        master,  # type: ignore[arg-type]
        text=text,
        command=command,  # type: ignore[arg-type]
        width=width,
        height=38,
        corner_radius=8,
        border_width=1,
        border_color=colors["field_border"],
        fg_color=colors["secondary"],
        hover_color=colors["secondary_hover"],
        text_color=colors["text"],
        font=font(14),
    )


def field_entry(master: object, colors: LightPalette, **kwargs: object) -> ctk.CTkEntry:
    return ctk.CTkEntry(
        master,  # type: ignore[arg-type]
        height=34,
        corner_radius=8,
        border_width=1,
        fg_color=colors["field"],
        border_color=colors["field_border"],
        text_color=colors["text"],
        placeholder_text_color=colors["muted"],
        font=font(14),
        **kwargs,  # type: ignore[arg-type]
    )
