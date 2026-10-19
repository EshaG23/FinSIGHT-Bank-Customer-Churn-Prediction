import plotly.graph_objects as go
import streamlit as st

import analytics
import config as cfg
import model_service as ms
import theme
import ui


def explain(what: str, reading: str) -> None:
    st.markdown(f'<div class="section-sub"><b>What it shows.</b> {what}<br><b>How to read it.</b> {reading}</div>', unsafe_allow_html=True)


def performance_tab() -> None:
    metrics = ui.get_metrics()
    if metrics is None:
        ui.empty_state("No evaluation metrics found", "Run compare_models.py to generate the comparison files.")
        return
    explain("Accuracy, precision, recall, F1 and ROC-AUC for each model on the held-out test set. Precision, recall and F1 are for the churn class.",
            "Higher is better. Look beyond accuracy: about 80% of customers stay, so accuracy alone flatters cautious models.")
    with ui.card("insights-metrics"):
        ui.plot(ui.metric_bars(metrics))
    shown = metrics[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Threshold"]].rename(
        columns={"ROC_AUC": "ROC-AUC", "Threshold": "Alert line"})
    table = shown.copy()
    for col in table.columns[1:]:
        table[col] = table[col].map(lambda v: f"{v:.3f}")
    st.markdown(ui.html_table(table, num_cols=list(table.columns[1:]),
                              best={c: "max" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}), unsafe_allow_html=True)
    st.caption("Bold marks the highest value per column. Alert lines differ by model, chosen on validation data only.")


def distribution_tab() -> None:
    df, preds = ui.get_dataset(), ui.get_predictions()
    split, _ = ui.get_split()
    explain("How many customers actually churned, and how many customers each model flags as likely churners on the test set.",
            "A model that flags far more customers than actually churned catches many leavers but raises many false alarms.")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        with ui.card("insights-donut"):
            if split is not None:
                churned = int(split.y.sum())
                ui.plot(ui.donut_churn({"retained": len(split) - churned, "churned": churned, "churn_rate": churned / len(split)},
                                       "Test set: retained vs churned"))
            elif df is not None:
                ui.plot(ui.donut_churn(analytics.overall_stats(df), "Dataset: retained vs churned"))
            else:
                ui.empty_state("Distribution unavailable", "Needs the dataset or the held-out test split.")
    with c2:
        with ui.card("insights-avp"):
            if preds:
                ui.plot(ui.actual_vs_predicted_fig(preds))
            else:
                ui.empty_state("Predictions unavailable", "Needs each model's held-out prediction export.", "activity")


def confusion_tab() -> None:
    metrics = ui.get_metrics()
    if metrics is None or not {"TN", "FP", "FN", "TP"}.issubset(metrics.columns):
        ui.empty_state("Confusion matrices unavailable", "They need the TN, FP, FN and TP counts from the comparison metrics.")
        return
    explain("For each model, how many test customers fall into each combination of actual and predicted outcome.",
            "The diagonal (top-left and bottom-right) holds correct predictions. Bottom-left are missed leavers, top-right are false alarms.")
    rows = list(metrics.itertuples(index=False))
    for start in range(0, len(rows), 3):
        cols = st.columns(3)
        for col, row in zip(cols, rows[start:start + 3]):
            with col:
                with ui.card(f"insights-cm-{start}-{row.Model.replace(' ', '')}"):
                    ui.plot(ui.confusion_fig(metrics[metrics.Model == row.Model].iloc[0], row.Model))
                    st.caption(f"Caught {row.TP} of {row.TP + row.FN} leavers. Raised {row.FP} false alarms.")


def roc_tab() -> None:
    if all(ms.roc_points(k) is None for k in cfg.MODEL_ORDER):
        ui.empty_state("ROC curves unavailable", "They need each model's held-out prediction file.")
        return
    explain("True positive rate against false positive rate as the alert line is moved from strict to lenient.",
            "A curve closer to the top-left corner is better; the dashed diagonal is guessing. AUC is the area under the curve.")
    with ui.card("insights-roc"):
        ui.plot(ui.roc_fig())
    st.caption("The SVM curve uses its decision margin as the ranking score. ROC-AUC does not depend on any alert line.")


def pr_tab() -> None:
    if all(ms.pr_points(k) is None for k in cfg.MODEL_ORDER):
        ui.empty_state("Precision-recall curves unavailable", "They need each model's held-out prediction file.")
        return
    explain("Precision against recall as the alert line moves. AP (average precision) summarises the curve.",
            "Higher and further right is better. With ~20% churners, the dashed baseline is the no-skill precision; "
            "this view is more revealing than ROC when the positive class is rare.")
    with ui.card("insights-pr"):
        ui.plot(ui.pr_fig())


def importance_tab(service: ms.ModelService) -> None:
    ui.callout("<b>Importance is not cause.</b> These charts show which inputs a model leans on when it scores customers. "
               "They do not show why customers leave, and correlated inputs can share or hide credit.")
    st.write("")
    options = service.available_keys()
    if not options:
        ui.empty_state("No models available", "Feature importance needs at least one loaded model.")
        return
    default = options.index("forest") if "forest" in options else 0
    key = st.selectbox("Model", options, index=default, format_func=lambda k: cfg.MODEL_SPECS[k].name, key="imp_model")
    table = ui.get_importance(key)
    if table is None:
        ui.empty_state("Importance unavailable", "It needs the saved test split in the outputs folder.", "activity")
        return
    c1, c2 = st.columns([1.3, 1])
    with c1:
        with ui.card("insights-importance"):
            ui.plot(ui.importance_bars(table, f"Permutation importance: {cfg.MODEL_SPECS[key].name}"))
    with c2:
        coeffs = ms.linear_coefficients(service, key)
        if coeffs is not None:
            p = theme.palette()
            ordered = coeffs.sort_values("Weight")
            fig = go.Figure(go.Bar(x=ordered["Weight"], y=ordered["Feature"], orientation="h",
                                   marker_color=[p["churned"] if w > 0 else p["retained"] for w in ordered["Weight"]],
                                   hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig.update_layout(title_text="Learned weights (standardised inputs)", xaxis_title="Weight", yaxis_title=None)
            with ui.card("insights-weights"):
                ui.plot(ui.style_fig(fig, 380, legend=False))
            st.caption("Positive weights push the score toward churn, negative weights away from it.")
        else:
            ui.callout("Tree models and KNN have no single weight per feature, so only permutation importance is shown for them. "
                       "Linear models (Logistic Regression, SVM) also get a weights chart.")
    st.caption("Permutation importance: drop in held-out ROC-AUC when one input is shuffled, averaged over 3 shuffles. "
               "Country and gender are shuffled as whole groups.")


def render() -> None:
    service = ui.get_service()
    ui.page_header("Analytics", "Model Performance & Insights",
                   "How the five models score on held-out customers, where they go wrong, and what each one relies on.")
    t = st.tabs(["Performance", "Churn distribution", "Confusion matrices", "ROC curves", "Precision-recall", "Feature importance"])
    with t[0]:
        performance_tab()
    with t[1]:
        distribution_tab()
    with t[2]:
        confusion_tab()
    with t[3]:
        roc_tab()
    with t[4]:
        pr_tab()
    with t[5]:
        importance_tab(service)


render()
