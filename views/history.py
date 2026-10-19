import streamlit as st

import state
import ui


def render() -> None:
    ui.page_header("Session log", "Prediction History",
                   "Every prediction from this browser session. Only a time, source, reference, model, outcome and score are kept. "
                   "New customer details are not stored.")
    history = state.history_frame()
    if history.empty:
        ui.empty_state("No predictions yet", "Generate a test record or predict a new customer and the results will be logged here.", "activity")
        return
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Export CSV", history.to_csv(index=False).encode("utf-8"), "finsight_prediction_history.csv",
                       "text/csv", width="stretch")
    if c2.button("Clear history", width="stretch", key="history_clear"):
        state.clear_history()
        st.rerun()
    shown = history.iloc[::-1].copy()
    shown["Churn score"] = shown["Churn score"].map(lambda v: "n/a" if v != v or v is None else f"{v:.3f}")
    shown["Actual"] = shown["Actual"].fillna("Unknown")
    ui.show_table(shown, mono_cols=["Churn score"])
    st.caption("History lives in session memory and resets when the browser session ends. The churn score is the model's raw output, "
               "not a calibrated probability, and is n/a for the SVM, which only has a margin. Actual is known only for test records.")


render()
