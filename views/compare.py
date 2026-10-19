import pandas as pd
import streamlit as st

import config as cfg
import model_service as ms
import state
import ui


def chosen_source():
    """Which customer to compare: the held-out test record, the new customer, or None when neither exists yet."""
    options = []
    if st.session_state.get("test_pick"):
        options.append(state.SOURCE_TEST)
    if st.session_state.get("all_results"):
        options.append(state.SOURCE_NEW)
    if not options:
        return None
    default = st.session_state.get("last_source") if st.session_state.get("last_source") in options else options[-1]
    if len(options) == 1:
        return options[0]
    return st.segmented_control("Customer to compare", options, default=default, key="compare_source") or default


def customer_tab() -> None:
    source = chosen_source()
    if source is None:
        ui.empty_state("No customer to compare yet",
                       "Pick a test record or enter a new customer first, and all five models will score the same customer here.", "users")
        st.write("")
        a, b, _ = st.columns([1.4, 1.4, 2.2])
        if a.button("Go to Test Dataset Explorer", type="primary", width="stretch", key="cmp_to_explorer"):
            ui.go_to("views/explorer.py")
        if b.button("Go to New Customer Prediction", width="stretch", key="cmp_to_predict"):
            ui.go_to("views/predict.py")
        return

    if source == state.SOURCE_TEST:
        pick = st.session_state["test_pick"]
        results, actual = pick["results"], pick["actual"]
        st.markdown(f'{ui.badge("Held-out test record", "accent", dot=False)} '
                    f'<span class="muted-text">#{pick["pos"]:,} · Customer ID {pick["customer_id"]} · actual outcome known: </span>{ui.actual_badge(actual)}',
                    unsafe_allow_html=True)
    else:
        results, actual = st.session_state["all_results"], None
        st.markdown(f'{ui.badge("New customer", "accent", dot=False)} '
                    f'<span class="muted-text">{ui.esc(st.session_state.get("customer_ref") or "")} · outcome unknown, so predictions are not marked correct or incorrect</span>',
                    unsafe_allow_html=True)

    ok, failed = ui.split_results(results)
    ui.section_title("This customer across all models", "Each model uses its own alert line, so raw scores are not directly comparable.")
    text, kind = ui.agreement_html(results, actual)
    ui.callout(text, kind)
    st.write("")
    ui.model_cards(results, actual)
    st.write("")
    st.markdown(ui.prediction_table(results, actual), unsafe_allow_html=True)
    st.caption("Churn probability is each model's raw output, not a calibrated probability. The SVM only has a decision margin.")
    for key, message in failed.items():
        st.warning(f"{cfg.MODEL_SPECS[key].name} is unavailable: {message}")
    rows = [{"key": r.model_key, "name": r.model_name, "pct": r.percentile, "thr": r.threshold_percentile, "churn": r.churn}
            for r in ok.values() if r.percentile is not None and r.threshold_percentile is not None]
    if rows:
        with ui.card("compare-rank"):
            ui.plot(ui.rank_chart(rows))


def performance_tab() -> None:
    metrics = ui.get_metrics()
    split, _ = ui.get_split()
    n_text = f"{len(split):,}" if split is not None else "the held-out"
    ui.section_title("Overall model performance",
                     f"Measured on all {n_text} held-out test customers the models never trained on. This is separate from any single-customer result.")
    if metrics is None:
        ui.empty_state("No evaluation metrics found", "Run compare_models.py to produce outputs/comparison/comparison_metrics.csv.")
        return
    shown = metrics[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Threshold"]].rename(
        columns={"ROC_AUC": "ROC-AUC", "Threshold": "Alert line"})
    table = shown.copy()
    for col in table.columns[1:]:
        table[col] = table[col].map(lambda v: f"{v:.3f}")
    st.markdown(ui.html_table(table, num_cols=list(table.columns[1:]),
                              best={c: "max" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}), unsafe_allow_html=True)
    st.caption("Bold marks the highest value in each metric column. Precision, recall and F1 are for the churn class "
               "(the positive class, Exited = 1). Alert lines differ by model and were chosen on validation data only.")
    with ui.card("compare-metrics"):
        ui.plot(ui.metric_bars(metrics, "Held-out test performance by model"))

    if {"TN", "FP", "FN", "TP"}.issubset(metrics.columns):
        ui.section_title("Confusion matrices", "How each model's predictions split into correct and incorrect on the test set.")
        rows = list(metrics.itertuples(index=False))
        for start in range(0, len(rows), 3):
            cols = st.columns(3)
            for col, row in zip(cols, rows[start:start + 3]):
                with col:
                    with ui.card(f"compare-cm-{start}-{row.Model.replace(' ', '')}"):
                        ui.plot(ui.confusion_fig(metrics[metrics.Model == row.Model].iloc[0], row.Model))
                        st.caption(f"Caught {row.TP} of {row.TP + row.FN} leavers. Raised {row.FP} false alarms.")

    ui.section_title("How to read these metrics")
    st.markdown(
        '<div class="churn-grid" style="margin-top:0">'
        "<div class='item'><b>Precision</b>Of the customers a model flags as churners, the share who really left.</div>"
        "<div class='item'><b>Recall</b>Of the customers who really left, the share the model caught.</div>"
        "<div class='item'><b>F1-score</b>A single number that balances precision and recall.</div>"
        "<div class='item'><b>ROC-AUC</b>How well the model ranks leavers above stayers, regardless of any alert line.</div>"
        "</div>", unsafe_allow_html=True)
    st.write("")
    leaders = []
    for col, label in [("ROC_AUC", "ROC-AUC"), ("F1", "F1"), ("Recall", "Recall"), ("Precision", "Precision"), ("Accuracy", "Accuracy")]:
        top = metrics.sort_values(col, ascending=False).iloc[0]
        leaders.append(f"<b>{label}</b>: {top.Model} ({top[col]:.3f})")
    ui.callout("<b>Which model is best?</b> It depends on the goal. " + ". ".join(leaders) + ".<br>"
               "Only about 20% of customers churn, so a model that predicts everyone stays would score near 80% accuracy and catch no "
               "leavers. Accuracy alone is a weak guide here. For churn, <b>recall</b> matters when missing a leaver is costly, "
               "but a model with high recall also raises more false alarms, and each false alarm can mean an unnecessary retention "
               "offer. Choose by recall when losing customers is the bigger cost, by precision when offers are expensive, "
               "and by ROC-AUC for overall ranking quality.")
    st.caption("Winners are descriptive on one test set, not proof of a real difference.")


def render() -> None:
    ui.page_header("Comparison", "Model Comparison",
                   "See where the five models agree and split on one customer, then check how each one performed across the whole held-out test set.")
    t1, t2 = st.tabs(["Individual customer", "Overall performance"])
    with t1:
        customer_tab()
    with t2:
        performance_tab()


render()
