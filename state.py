"""Session-state helpers: theme, current customer sources, last results and prediction history.

History keeps only a reference, time, source, model, outcome and score. New
customer inputs are never written to the history table.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

import config as cfg
import model_service as ms
import theme

HISTORY_COLUMNS = ["Time (UTC)", "Source", "Reference", "Model", "Prediction", "Churn score", "Actual"]
SOURCE_TEST = "Test record"
SOURCE_NEW = "New customer"


def init() -> None:
    st.session_state.setdefault("theme", theme.DEFAULT_MODE)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("customer", None)       # validated new-customer dict (this session only)
    st.session_state.setdefault("all_results", None)    # new customer: key -> PredictionResult | error string
    st.session_state.setdefault("customer_ref", None)
    st.session_state.setdefault("test_pick", None)      # {"pos", "customer_id", "actual", "results"}
    st.session_state.setdefault("last_source", None)    # SOURCE_TEST | SOURCE_NEW


def _row(ref: str, source: str, result: ms.PredictionResult, actual: int | None) -> dict[str, Any]:
    return {
        "Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source, "Reference": ref, "Model": result.model_name, "Prediction": result.label,
        # SVM margins are not probabilities, so no score is recorded for them.
        "Churn score": round(result.score, 4) if result.has_probability_scale else None,
        "Actual": None if actual is None else ms.TARGET_LABELS[actual],
    }


def record_run(ref: str, source: str, results: dict[str, Any], actual: int | None = None) -> None:
    """Log one row per model that produced a result for this customer."""
    for key in cfg.MODEL_ORDER:
        result = results.get(key)
        if isinstance(result, ms.PredictionResult):
            st.session_state["history"].append(_row(ref, source, result, actual))


def new_reference() -> str:
    return "FS-" + uuid.uuid4().hex[:6].upper()


def history_frame() -> pd.DataFrame:
    return pd.DataFrame(st.session_state["history"], columns=HISTORY_COLUMNS)


def clear_history() -> None:
    st.session_state["history"] = []


def reset_customer() -> None:
    st.session_state["customer"] = None
    st.session_state["all_results"] = None
    st.session_state["customer_ref"] = None
    if st.session_state.get("last_source") == SOURCE_NEW:
        st.session_state["last_source"] = SOURCE_TEST if st.session_state.get("test_pick") else None
