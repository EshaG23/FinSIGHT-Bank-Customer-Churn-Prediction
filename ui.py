"""Shared UI pieces: theme controls, CSS injection, HTML components, cached loaders, themed Plotly charts.

Everything colour-related is read from ``theme`` at call time, so a change of
mode restyles HTML components *and* every chart on the next rerun.
"""
from __future__ import annotations

import html
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics
import config as cfg
import model_service as ms
import state
import theme

# Lucide-style 24x24 stroke icons (inline SVG, no emoji).
_ICONS = {
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "trend-down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "award": '<circle cx="12" cy="8" r="6"/><path d="M15.5 13.2 17 22l-5-3-5 3 1.5-8.8"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "inbox": '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "x": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "check-circle": '<circle cx="12" cy="12" r="10"/><polyline points="8 12.5 11 15.5 16 9.5"/>',
    "x-circle": '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
    "stay": '<path d="M20 6 9 17l-5-5"/>',
    "leave": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
}


def icon(name: str, size: int = 16) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>')


def esc(text: Any) -> str:
    return html.escape(str(text))


# ---------------------------------------------------------------------------
# Theme controls and page chrome
# ---------------------------------------------------------------------------
def _theme_from_sidebar_toggle() -> None:
    theme.set_mode(theme.DARK if st.session_state.get("theme_toggle_sidebar") else theme.LIGHT)


def _flip_theme() -> None:
    theme.set_mode(theme.LIGHT if theme.current() == theme.DARK else theme.DARK)


def inject_css() -> None:
    st.markdown(f"<style>{theme.build_css(theme.current())}</style>", unsafe_allow_html=True)


def theme_toggle_sidebar() -> None:
    """Dark-mode switch in the sidebar. The stored mode is the single source of truth."""
    st.session_state["theme_toggle_sidebar"] = theme.current() == theme.DARK
    st.toggle("Dark mode", key="theme_toggle_sidebar", on_change=_theme_from_sidebar_toggle,
              help="Switch the whole interface between Light and Dark mode.")


def theme_button_topbar() -> None:
    """Compact switch in the page header, so it stays reachable when the sidebar is collapsed."""
    dark = theme.current() == theme.DARK
    st.button("Light mode" if dark else "Dark mode", key="theme_button_top", on_click=_flip_theme,
              icon=":material/light_mode:" if dark else ":material/dark_mode:", width="stretch",
              help="Switch between Light and Dark mode.")


def sidebar_brand(service: ms.ModelService) -> None:
    ready = len(service.available_keys())
    st.markdown(
        f'<div class="brand"><div class="brand-mark">{icon("shield", 18)}</div>'
        f'<div><div class="brand-name">FinSIGHT</div><div class="brand-sub">Churn intelligence</div></div></div>',
        unsafe_allow_html=True)
    st.markdown('<div class="side-label">Appearance</div>', unsafe_allow_html=True)
    theme_toggle_sidebar()
    st.markdown('<div class="side-label">Navigate</div>', unsafe_allow_html=True)


def sidebar_footer(service: ms.ModelService) -> None:
    ready = len(service.available_keys())
    st.markdown(
        f'<div class="side-foot"><b>{ready} of {len(cfg.MODEL_ORDER)}</b> models ready<br>'
        'Predictions come from the saved project models. Scores are not calibrated probabilities.</div>',
        unsafe_allow_html=True)


def card(key: str):
    """A themed card container. The key must be unique per page, and stable across reruns."""
    return st.container(key=f"card-{key}")


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{esc(eyebrow)}</div><div class="page-title">{esc(title)}</div>'
                f'<div class="page-sub">{esc(subtitle)}</div>', unsafe_allow_html=True)


def section_title(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="section-title">{esc(title)}</div>'
                + (f'<div class="section-sub">{esc(sub)}</div>' if sub else ""), unsafe_allow_html=True)


def kpi_card(label: str, value: str, caption: str, icon_name: str, tone: str = "", small: bool = False) -> str:
    return (f'<div class="card kpi"><div class="kpi-top"><span class="kpi-label">{esc(label)}</span>'
            f'<span class="kpi-icon {tone}">{icon(icon_name, 16)}</span></div>'
            f'<div class="kpi-value{" small" if small else ""}">{esc(value)}</div>'
            f'<div class="kpi-cap">{esc(caption)}</div></div>')


def empty_state(title: str, text: str, icon_name: str = "inbox") -> None:
    st.markdown(f'<div class="empty"><div class="ic">{icon(icon_name, 22)}</div>'
                f'<div class="t">{esc(title)}</div><div class="d">{esc(text)}</div></div>', unsafe_allow_html=True)


def callout(text_html: str, kind: str = "") -> None:
    st.markdown(f'<div class="callout {kind}">{text_html}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral", dot: bool = True, icon_name: str | None = None) -> str:
    lead = icon(icon_name, 13) if icon_name else ("<span class=dot></span>" if dot else "")
    return f'<span class="badge {kind}">{lead}{esc(text)}</span>'


TIER_CLASS = {"Low": "low", "Moderate": "mid", "High": "high"}


def tier_badge(tier: str) -> str:
    return badge(f"{tier} risk", TIER_CLASS[tier])


def go_to(page: str) -> None:
    st.switch_page(page)


def friendly_error(prefix: str, exc: Exception) -> None:
    """Show a readable message to the user. Technical details go to the server log."""
    import logging
    logging.getLogger("finsight").exception("%s", prefix, exc_info=exc)
    st.error(f"{prefix} {exc}")


# ---------------------------------------------------------------------------
# Outcome vocabulary (labels are mapped from the verified encoding 0 = stayed, 1 = churned)
# ---------------------------------------------------------------------------
def outcome_label(churn: bool | int) -> str:
    return ms.TARGET_LABELS[int(bool(churn))]


def predicted_badge(result: ms.PredictionResult) -> str:
    if result.churn:
        return badge("Likely to Churn", "high", icon_name="leave")
    return badge("Likely to Stay", "low", icon_name="stay")


def actual_badge(actual: int) -> str:
    return (badge("Churned", "high", icon_name="leave") if actual == 1 else badge("Stayed", "low", icon_name="stay"))


def verdict_badge(correct: bool) -> str:
    return (badge("Correct", "good", icon_name="check-circle") if correct
            else badge("Incorrect", "bad", icon_name="x-circle"))


def score_text(r: ms.PredictionResult) -> str:
    return f"{r.score * 100:.1f}%" if r.has_probability_scale else f"{r.score:+.2f} margin"


def threshold_text(r: ms.PredictionResult) -> str:
    return f"{r.threshold * 100:.1f}%" if r.has_probability_scale else f"{r.threshold:+.2f}"


def model_color(key: str) -> str:
    return theme.model_colors()[key]


# ---------------------------------------------------------------------------
# HTML table (st.dataframe draws on a canvas that CSS cannot theme, so tables are HTML)
# ---------------------------------------------------------------------------
def html_table(df: pd.DataFrame, html_cols: Iterable[str] = (), num_cols: Iterable[str] = (),
               mono_cols: Iterable[str] = (), best: dict[str, str] | None = None) -> str:
    """Render a DataFrame as a themed table. Columns in ``html_cols`` are trusted pre-rendered HTML;
    everything else is escaped. ``best`` maps a numeric column to 'max' or 'min' to emphasise the leader."""
    html_cols, num_cols, mono_cols = set(html_cols), set(num_cols), set(mono_cols)
    leaders: dict[str, Any] = {}
    for col, how in (best or {}).items():
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            leaders[col] = values.max() if how == "max" else values.min()
    head = "".join(f'<th class="{"num" if c in num_cols else ""}">{esc(c)}</th>' for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            value = row[col]
            classes = ["num"] if col in num_cols else []
            if col in mono_cols:
                classes.append("mono")
            if col in leaders and pd.notna(pd.to_numeric(value, errors="coerce")) and np.isclose(float(value), leaders[col]):
                classes.append("best")
            text = value if col in html_cols else esc("" if pd.isna(value) else value)
            cells += f'<td class="{" ".join(classes)}">{text}</td>'
        body += f"<tr>{cells}</tr>"
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def show_table(df: pd.DataFrame, **kwargs: Any) -> None:
    st.markdown(html_table(df, **kwargs), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models")
def get_service() -> ms.ModelService:
    return ms.ModelService()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame | None:
    return ms.load_dataset()


@st.cache_data(show_spinner=False)
def get_metrics() -> pd.DataFrame | None:
    return ms.load_metrics()


@st.cache_data(show_spinner=False)
def get_rates() -> pd.DataFrame | None:
    df = get_dataset()
    return analytics.segment_rates(df) if df is not None else None


@st.cache_data(show_spinner=False)
def get_split() -> tuple[ms.TestSplit | None, str | None]:
    """(split, None) when the held-out split is present and verified, else (None, plain-language reason)."""
    try:
        return ms.load_test_split(), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def get_predictions() -> dict[str, pd.DataFrame]:
    return {k: p for k in cfg.MODEL_ORDER if (p := ms.load_predictions(k)) is not None}


@st.cache_data(show_spinner="Measuring feature importance")
def get_importance(key: str) -> pd.DataFrame | None:
    return ms.permutation_importance(get_service(), key)


def best_model_summary(metrics: pd.DataFrame | None) -> dict[str, str] | None:
    """Pick the headline model on ROC-AUC (threshold-free, robust to class imbalance).

    Accuracy alone is misleading here because ~80% of customers stay. The
    caption names the leaders on other metrics so the trade-offs stay visible.
    """
    if metrics is None or metrics.empty:
        return None
    top = metrics.sort_values("ROC_AUC", ascending=False).iloc[0]
    recall_leader = metrics.sort_values("Recall", ascending=False).iloc[0]
    f1_leader = metrics.sort_values("F1", ascending=False).iloc[0]
    parts = [f"Highest ROC-AUC ({top.ROC_AUC:.3f})"]
    if recall_leader.Model != top.Model:
        parts.append(f"{recall_leader.Model} leads recall")
    if f1_leader.Model != top.Model:
        parts.append(f"{f1_leader.Model} leads F1")
    return {"name": str(top.Model), "caption": ". ".join(parts) + "."}


# ---------------------------------------------------------------------------
# Prediction renderers shared by the Explorer, New Customer and Compare pages
# ---------------------------------------------------------------------------
def split_results(results: dict[str, Any]) -> tuple[dict[str, ms.PredictionResult], dict[str, str]]:
    ok = {k: v for k, v in results.items() if isinstance(v, ms.PredictionResult)}
    failed = {k: v for k, v in results.items() if not isinstance(v, ms.PredictionResult)}
    return ok, failed


def agreement_html(results: dict[str, Any], actual: int | None = None) -> tuple[str, str]:
    """(html, callout kind) describing agreement/disagreement among the models that produced results."""
    ok, _ = split_results(results)
    n = len(ok)
    if n == 0:
        return "<b>No model produced a prediction.</b>", "bad"
    flagged = [r.model_name for r in ok.values() if r.churn]
    stayed = [r.model_name for r in ok.values() if not r.churn]
    qualifier = "" if n == len(cfg.MODEL_ORDER) else f" (only {n} of {len(cfg.MODEL_ORDER)} models are available)"
    if not flagged or not stayed:
        word = "Likely to Churn" if flagged else "Likely to Stay"
        text = f"<b>All {n} models agree{qualifier}:</b> {word}."
        if actual is not None:
            right = (flagged and actual == 1) or (stayed and actual == 0)
            text += (" This matches the known outcome." if right
                     else f" The known outcome is <b>{ms.TARGET_LABELS[actual]}</b>, so they were all wrong on this record.")
        return text, ("bad" if flagged else "good")
    text = (f"<b>The models disagree{qualifier}.</b> {len(flagged)} predict churn ({esc(', '.join(flagged))}); "
            f"{len(stayed)} predict the customer stays ({esc(', '.join(stayed))}). "
            "Different algorithms pick up different patterns, and each applies its own alert line, so a split is useful "
            "information about a borderline profile rather than an application error.")
    if actual is not None:
        wrong = flagged if actual == 0 else stayed
        text += f" The known outcome is <b>{ms.TARGET_LABELS[actual]}</b>, so {esc(', '.join(wrong))} got this record wrong."
    return text, "warn"


def prediction_table(results: dict[str, Any], actual: int | None = None) -> str:
    """Comparison table. Correct/Incorrect appears only when the true outcome is known."""
    rows = []
    for key in cfg.MODEL_ORDER:
        value = results.get(key)
        swatch = f'<span class="model-swatch" style="display:inline-block;background:{model_color(key)};margin-right:8px"></span>'
        name = f"{swatch}<b>{esc(cfg.MODEL_SPECS[key].name)}</b>"
        if not isinstance(value, ms.PredictionResult):
            row = {"Model": name, "Predicted outcome": badge("Unavailable", "neutral", dot=False),
                   "Churn probability": esc(value or "Model not loaded")[:90], "Alert line": "n/a"}
            if actual is not None:
                row.update({"Actual outcome": actual_badge(actual), "Result": badge("Not evaluated", "neutral", dot=False)})
        else:
            prob = f"{value.score * 100:.1f}%" if value.has_probability_scale else f"n/a (margin {value.score:+.2f})"
            row = {"Model": name, "Predicted outcome": predicted_badge(value), "Churn probability": prob,
                   "Alert line": threshold_text(value)}
            if actual is not None:
                row.update({"Actual outcome": actual_badge(actual), "Result": verdict_badge(value.churn == bool(actual))})
        rows.append(row)
    df = pd.DataFrame(rows)
    return html_table(df, html_cols=[c for c in df.columns if c not in ("Churn probability", "Alert line")],
                      mono_cols=["Churn probability", "Alert line"])


def model_cards(results: dict[str, Any], actual: int | None = None) -> None:
    ok, _ = split_results(results)
    n = len(ok)
    n_flag = sum(r.churn for r in ok.values())
    majority_churn = n_flag * 2 >= n if n else False
    cols = st.columns(len(cfg.MODEL_ORDER))
    for col, key in zip(cols, cfg.MODEL_ORDER):
        value = results.get(key)
        with col:
            swatch = f'<span class="model-swatch" style="background:{model_color(key)}"></span>'
            name = f'<div class="model-name">{swatch}{esc(cfg.MODEL_SPECS[key].name)}</div>'
            if isinstance(value, ms.PredictionResult):
                disagree = actual is None and n > 1 and 0 < n_flag < n and value.churn != majority_churn
                css = ""
                verdict = ""
                if actual is not None:
                    right = value.churn == bool(actual)
                    css = "right" if right else "wrong"
                    verdict = verdict_badge(right)
                elif disagree:
                    css = "disagree"
                    verdict = badge("Disagrees with majority", "mid", dot=False)
                score_label = "Churn probability" if value.has_probability_scale else "Decision margin"
                st.markdown(
                    f'<div class="card model-card {css}">{name}<div>{predicted_badge(value)}</div>'
                    f'<div><div class="model-line">{score_label}</div><div class="model-score">{score_text(value)}</div></div>'
                    f'<div class="model-line">Alert line {threshold_text(value)}</div>{verdict}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card model-card off">{name}<div>{badge("Unavailable", "neutral")}</div>'
                            f'<div class="model-line">{esc(value or "Model not loaded")}</div></div>', unsafe_allow_html=True)


def fields_html(fields: list[tuple[str, str]]) -> str:
    tiles = "".join(f'<div class="field"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>' for k, v in fields)
    return f'<div class="fields">{tiles}</div>'


def customer_fields(raw: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Age", f"{int(raw['Age'])}"), ("Gender", str(raw["Gender"])), ("Geography", str(raw["Geography"])),
        ("Credit score", f"{int(raw['CreditScore'])}"), ("Balance", f"{float(raw['Balance']):,.2f}"),
        ("Estimated salary", f"{float(raw['EstimatedSalary']):,.2f}"), ("Tenure", f"{int(raw['Tenure'])} yr"),
        ("Products", f"{int(raw['NumOfProducts'])}"), ("Credit card", "Yes" if int(raw["HasCrCard"]) else "No"),
        ("Active member", "Yes" if int(raw["IsActiveMember"]) else "No"),
    ]


# ---------------------------------------------------------------------------
# Result meter (HTML so it can animate)
# ---------------------------------------------------------------------------
def meter_html(result: ms.PredictionResult) -> str:
    kind = TIER_CLASS[result.risk_tier]
    if result.has_probability_scale:
        fill = max(2.0, min(100.0, result.score * 100))
        tick = max(0.0, min(100.0, result.threshold * 100))
        left = f"Model churn score <b>{result.score * 100:.1f}%</b>"
        right = f"Alert line {result.threshold * 100:.1f}%"
        scale = "<span>0%</span><span>50%</span><span>100%</span>"
    else:
        pct = result.percentile if result.percentile is not None else 0.0
        fill = max(2.0, min(100.0, pct))
        tick = result.threshold_percentile if result.threshold_percentile is not None else 0.0
        left = f"Decision margin <b>{result.score:+.2f}</b> (ranking score)"
        right = f"Alert line {result.threshold:+.2f}"
        scale = "<span>Lowest risk rank</span><span>Highest risk rank</span>"
    return (f'<div class="meter"><div class="meter-label"><span>{left}</span><span>{right}</span></div>'
            f'<div class="meter-track"><div class="meter-fill {kind}" style="--w:{fill:.1f}%"></div>'
            f'<div class="meter-tick" style="--t:{tick:.1f}%"><span>alert line</span></div></div>'
            f'<div class="meter-scale">{scale}</div></div>')


# ---------------------------------------------------------------------------
# Plotly (all styling comes from theme.py so both modes stay readable)
# ---------------------------------------------------------------------------
def plot(fig: go.Figure, key: str | None = None) -> None:
    """Render a figure with our own styling; Streamlit's chart theme is disabled so it cannot override it."""
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False}, key=key)


def style_fig(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    p = theme.palette()
    fig.update_layout(
        template="none", height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=theme.FONT_STACK, color=p["text"], size=12),
        showlegend=legend, legend=dict(orientation="h", y=-0.22, x=0, font=dict(color=p["muted"], size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=p["hover_bg"], bordercolor=p["border_strong"], font=dict(color=p["text"], family=theme.FONT_STACK)),
        title=dict(font=dict(size=14, color=p["text"]), x=0.0, xanchor="left"),
        colorway=list(theme.model_colors().values()),
    )
    axis = dict(gridcolor=p["plot_grid"], zerolinecolor=p["plot_axis"], linecolor=p["plot_axis"],
                tickfont=dict(color=p["muted"], size=12), title_font=dict(color=p["muted"], size=12))
    fig.update_xaxes(automargin=True, **axis)
    fig.update_yaxes(automargin=True, **axis)
    return fig


def _outcome_colors() -> tuple[str, str]:
    p = theme.palette()
    return p["retained"], p["churned"]


def donut_churn(stats: dict[str, float], title: str = "Retained vs churned customers") -> go.Figure:
    p = theme.palette()
    retained, churned = _outcome_colors()
    fig = go.Figure(go.Pie(
        labels=["Retained", "Churned"], values=[stats["retained"], stats["churned"]], hole=0.68, sort=False,
        marker=dict(colors=[retained, churned], line=dict(color=p["surface"], width=3)),
        textinfo="percent", textfont=dict(color="#FFFFFF" if theme.current() == theme.LIGHT else "#0A0F1F", size=13),
        hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b>{stats['churn_rate'] * 100:.1f}%</b><br><span style='font-size:11px'>churn rate</span>",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=22, color=p["text"]))
    return style_fig(fig, 320).update_layout(title_text=title)


def segment_bars(table: pd.DataFrame, title: str) -> go.Figure:
    retained, churned = _outcome_colors()
    fig = go.Figure()
    fig.add_bar(x=table["Segment"].astype(str), y=table["Retained"], name="Retained", marker_color=retained,
                hovertemplate="%{x}<br>Retained: %{y:,}<extra></extra>")
    fig.add_bar(x=table["Segment"].astype(str), y=table["Churned"], name="Churned", marker_color=churned,
                customdata=(table["ChurnRate"] * 100),
                hovertemplate="%{x}<br>Churned: %{y:,} (%{customdata:.1f}% of segment)<extra></extra>")
    fig.update_layout(barmode="stack", title_text=title, yaxis_title="Customers")
    return style_fig(fig, 320)


METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]


def metric_bars(metrics: pd.DataFrame, title: str = "Held-out test performance") -> go.Figure:
    """Grouped bars: one colour per MODEL (same colour everywhere), metrics on the x axis."""
    by_name = {cfg.MODEL_SPECS[k].name: k for k in cfg.MODEL_ORDER}
    fig = go.Figure()
    for _, row in metrics.iterrows():
        key = by_name.get(row["Model"])
        fig.add_bar(x=[c.replace("_", "-") for c in METRIC_COLUMNS], y=[row[c] for c in METRIC_COLUMNS], name=row["Model"],
                    marker_color=model_color(key) if key else None,
                    hovertemplate=f"{row['Model']}<br>%{{x}}: %{{y:.3f}}<extra></extra>")
    fig.update_layout(barmode="group", title_text=title, yaxis=dict(range=[0, 1], title=None), xaxis_title=None)
    return style_fig(fig, 380)


def actual_vs_predicted_fig(preds: dict[str, pd.DataFrame]) -> go.Figure:
    """Count of customers each model flagged as churners next to the actual churner count."""
    p = theme.palette()
    names, flagged, key_list = [], [], []
    for key in cfg.MODEL_ORDER:
        if key in preds:
            names.append(cfg.MODEL_SPECS[key].name)
            flagged.append(int(preds[key].y_pred.sum()))
            key_list.append(key)
    actual = int(next(iter(preds.values())).y_true.sum())
    fig = go.Figure()
    fig.add_bar(x=["Actual churners"], y=[actual], name="Actual", marker_color=p["churned"],
                text=[f"{actual:,}"], textposition="outside", textfont=dict(color=p["text"]),
                hovertemplate="Actual churners in the test set: %{y:,}<extra></extra>")
    for name, count, key in zip(names, flagged, key_list):
        fig.add_bar(x=[name], y=[count], name=name, marker_color=model_color(key), text=[f"{count:,}"],
                    textposition="outside", textfont=dict(color=p["text"]),
                    hovertemplate=f"{name} flagged %{{y:,}} customers as likely churners<extra></extra>")
    fig.update_layout(title_text="Actual churners vs customers each model flags", yaxis_title="Customers", showlegend=False)
    fig.update_yaxes(range=[0, max([actual] + flagged) * 1.15])
    return style_fig(fig, 340, legend=False)


def importance_bars(table: pd.DataFrame, title: str) -> go.Figure:
    p = theme.palette()
    ordered = table.sort_values("Importance")
    fig = go.Figure(go.Bar(
        x=ordered["Importance"], y=ordered["Feature"], orientation="h",
        marker=dict(color=ordered["Importance"], colorscale=[[0, p["accent_2"]], [1, p["accent"]]]),
        error_x=dict(type="data", array=ordered["Spread"], color=p["muted"], thickness=1),
        hovertemplate="%{y}<br>ROC-AUC drop: %{x:.4f}<extra></extra>"))
    fig.update_layout(title_text=title, xaxis_title="Drop in ROC-AUC when shuffled", yaxis_title=None)
    return style_fig(fig, 380, legend=False)


def rank_chart(rows: list[dict[str, Any]]) -> go.Figure:
    """Where the customer ranks among held-out customers, per model, with each model's alert line."""
    p = theme.palette()
    fig = go.Figure()
    names = [r["name"] for r in rows]
    fig.add_bar(y=names, x=[r["pct"] for r in rows], orientation="h", name="This customer's rank", showlegend=False,
                marker_color=[model_color(r["key"]) for r in rows],
                text=[("Flags churn" if r["churn"] else "Not flagged") for r in rows], textposition="inside",
                insidetextanchor="end", textfont=dict(color="#FFFFFF" if theme.current() == theme.LIGHT else "#0A0F1F", size=12),
                hovertemplate="%{y}<br>Scores higher than %{x:.0f}% of held-out customers<extra></extra>")
    fig.add_trace(go.Scatter(y=names, x=[r["thr"] for r in rows], mode="markers", name="Model's alert line",
                             marker=dict(symbol="line-ns-open", size=26, color=p["text"], line=dict(width=3, color=p["text"])),
                             hovertemplate="%{y}<br>Alert line sits at the %{x:.0f}th percentile<extra></extra>"))
    fig.update_layout(title_text="Customer risk rank vs each model's alert line",
                      xaxis=dict(range=[0, 100], title="Percentile among 2,000 held-out customers (higher = riskier)"),
                      yaxis=dict(autorange="reversed", title=None))
    return style_fig(fig, 340)


def confusion_fig(row: pd.Series, name: str) -> go.Figure:
    p = theme.palette()
    z = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted stayed", "Predicted churned"], y=["Actually stayed", "Actually churned"],
        colorscale=[[0, p["surface_2"]], [1, p["accent"]]], showscale=False,
        hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>", xgap=3, ygap=3))
    top = z.max()
    for i, y_name in enumerate(["Actually stayed", "Actually churned"]):
        for j, x_name in enumerate(["Predicted stayed", "Predicted churned"]):
            dark_cell = z[i, j] / top > 0.5
            fig.add_annotation(x=x_name, y=y_name, text=f"<b>{int(z[i, j]):,}</b>", showarrow=False,
                               font=dict(size=17, color=p["accent_fg"] if dark_cell else p["text"]))
    fig.update_layout(title_text=name)
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, 290, legend=False)


def roc_fig() -> go.Figure:
    p = theme.palette()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color=p["subtle"], dash="dash", width=1), hoverinfo="skip"))
    for key in cfg.MODEL_ORDER:
        points = ms.roc_points(key)
        if points is None:
            continue
        fpr, tpr, auc = points
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{cfg.MODEL_SPECS[key].name} (AUC {auc:.3f})",
                                 line=dict(color=model_color(key), width=2.4),
                                 hovertemplate=f"{cfg.MODEL_SPECS[key].name}<br>FPR %{{x:.2f}}<br>TPR %{{y:.2f}}<extra></extra>"))
    fig.update_layout(title_text="ROC curves on the held-out test set", xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    return style_fig(fig, 460)


def pr_fig() -> go.Figure:
    p = theme.palette()
    preds = get_predictions()
    fig = go.Figure()
    if preds:
        base = float(next(iter(preds.values())).y_true.mean())
        fig.add_trace(go.Scatter(x=[0, 1], y=[base, base], mode="lines", name=f"No-skill baseline ({base:.2f})",
                                 line=dict(color=p["subtle"], dash="dash", width=1), hoverinfo="skip"))
    for key in cfg.MODEL_ORDER:
        points = ms.pr_points(key)
        if points is None:
            continue
        recall, precision, ap = points
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{cfg.MODEL_SPECS[key].name} (AP {ap:.3f})",
                                 line=dict(color=model_color(key), width=2.4),
                                 hovertemplate=f"{cfg.MODEL_SPECS[key].name}<br>Recall %{{x:.2f}}<br>Precision %{{y:.2f}}<extra></extra>"))
    fig.update_layout(title_text="Precision-recall curves on the held-out test set", xaxis_title="Recall",
                      yaxis_title="Precision", yaxis_range=[0, 1.02])
    return style_fig(fig, 460)
