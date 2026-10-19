"""Central theme configuration for FinSIGHT: design tokens for Light and Dark mode.

Every colour used by the CSS (assets/styles.css) and by the Plotly charts comes
from the two palettes below, so a theme switch changes the whole interface
consistently. The selected mode lives in ``st.session_state["theme"]`` and is
therefore kept across reruns and page changes within a browser session.

The module has no Streamlit dependency except ``current()``/``set_mode()``,
which means the palettes and the contrast helper can be unit tested directly.
"""
from __future__ import annotations

from pathlib import Path

LIGHT = "light"
DARK = "dark"
MODES = (LIGHT, DARK)
DEFAULT_MODE = LIGHT

PALETTES: dict[str, dict[str, str]] = {
    LIGHT: {
        # surfaces
        "bg": "#F6F7FB", "bg_glow_1": "rgba(79,70,229,.07)", "bg_glow_2": "rgba(37,99,235,.05)",
        "surface": "#FFFFFF", "surface_2": "#F1F3F9", "sidebar": "#FFFFFF",
        "border": "#DDE1EA", "border_strong": "#BCC3D3",
        # text
        "text": "#0F172A", "muted": "#475569", "subtle": "#64748B",
        # brand
        "accent": "#4F46E5", "accent_hover": "#4338CA", "accent_fg": "#FFFFFF",
        "accent_soft": "rgba(79,70,229,.09)", "accent_text": "#4338CA", "accent_2": "#2563EB",
        # status (fg = text colour, bg = tinted fill, bd = border)
        "good_fg": "#036C4A", "good_bg": "#E7F7EF", "good_bd": "#9FDCC0",
        "bad_fg": "#B4232C", "bad_bg": "#FDECEE", "bad_bd": "#F3B0B6",
        "warn_fg": "#8A4B00", "warn_bg": "#FFF4DE", "warn_bd": "#EBC77E",
        # charts
        "plot_grid": "rgba(15,23,42,.09)", "plot_axis": "rgba(15,23,42,.28)",
        "hover_bg": "#FFFFFF", "shadow": "0 1px 2px rgba(15,23,42,.05), 0 8px 24px rgba(15,23,42,.06)",
        "shadow_hover": "0 2px 4px rgba(15,23,42,.06), 0 14px 32px rgba(15,23,42,.10)",
        "input_bg": "#FFFFFF", "track": "#E4E8F1",
        "retained": "#0A7C7E", "churned": "#D1411F",
    },
    DARK: {
        "bg": "#0A0F1F", "bg_glow_1": "rgba(124,108,255,.14)", "bg_glow_2": "rgba(77,163,255,.08)",
        "surface": "#121A31", "surface_2": "#18224A", "sidebar": "#0D1428",
        "border": "#2A3660", "border_strong": "#3E4D82",
        "text": "#F3F5FC", "muted": "#B2BCD9", "subtle": "#94A0C4",
        "accent": "#5846E3", "accent_hover": "#6A59F0", "accent_fg": "#FFFFFF",
        "accent_soft": "rgba(124,108,255,.18)", "accent_text": "#C4BEFF", "accent_2": "#60A5FA",
        "good_fg": "#4ADE9F", "good_bg": "rgba(52,211,153,.12)", "good_bd": "rgba(52,211,153,.38)",
        "bad_fg": "#FF8A96", "bad_bg": "rgba(251,113,133,.13)", "bad_bd": "rgba(251,113,133,.42)",
        "warn_fg": "#FBC25B", "warn_bg": "rgba(251,191,36,.12)", "warn_bd": "rgba(251,191,36,.40)",
        "plot_grid": "rgba(178,188,217,.16)", "plot_axis": "rgba(178,188,217,.38)",
        "hover_bg": "#1E2A55", "shadow": "0 1px 2px rgba(0,0,0,.35), 0 10px 30px rgba(0,0,0,.35)",
        "shadow_hover": "0 2px 4px rgba(0,0,0,.4), 0 16px 36px rgba(0,0,0,.45)",
        "input_bg": "#0D1428", "track": "#222E5C",
        "retained": "#2DD4BF", "churned": "#FF8A5B",
    },
}

# One colour per model, identical in every chart so a model keeps its identity.
MODEL_COLORS: dict[str, dict[str, str]] = {
    LIGHT: {"knn": "#2563EB", "logistic": "#B84E0A", "tree": "#0B7A5E", "forest": "#7C3AED", "svm": "#BE185D"},
    DARK: {"knn": "#60A5FA", "logistic": "#FBA24E", "tree": "#34D399", "forest": "#A78BFA", "svm": "#F472B6"},
}

FONT_STACK = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO_STACK = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

_CSS_FILE = Path(__file__).resolve().parent / "assets" / "styles.css"


# ---------------------------------------------------------------------------
# Mode selection (session state)
# ---------------------------------------------------------------------------
def current() -> str:
    """The active mode. Falls back to the default outside a Streamlit session (tests, scripts)."""
    try:
        import streamlit as st
        mode = st.session_state.get("theme", DEFAULT_MODE)
    except Exception:  # pragma: no cover - streamlit always importable in the app
        mode = DEFAULT_MODE
    return mode if mode in MODES else DEFAULT_MODE


def set_mode(mode: str) -> None:
    import streamlit as st
    st.session_state["theme"] = mode if mode in MODES else DEFAULT_MODE


def palette(mode: str | None = None) -> dict[str, str]:
    return PALETTES[mode or current()]


def model_colors(mode: str | None = None) -> dict[str, str]:
    return MODEL_COLORS[mode or current()]


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def css_variables(mode: str) -> str:
    pal = PALETTES[mode]
    lines = "".join(f"--{key.replace('_', '-')}:{value};" for key, value in pal.items())
    return (f":root{{{lines}--font:{FONT_STACK};--mono:{MONO_STACK};--radius:14px;"
            f"color-scheme:{mode};}}")


def build_css(mode: str | None = None) -> str:
    """Variable block for the chosen mode followed by the shared stylesheet."""
    mode = mode or current()
    return css_variables(mode) + "\n" + _CSS_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Accessibility helper (used by tests and the visual audit)
# ---------------------------------------------------------------------------
def _channel(value: float) -> float:
    value /= 255
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def _rgb(colour: str) -> tuple[float, float, float]:
    colour = colour.strip().lstrip("#")
    if len(colour) == 3:
        colour = "".join(ch * 2 for ch in colour)
    return tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2.x contrast ratio between two #RRGGBB colours."""
    def luminance(colour: str) -> float:
        r, g, b = (_channel(c) for c in _rgb(colour))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    a, b = luminance(foreground), luminance(background)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)
