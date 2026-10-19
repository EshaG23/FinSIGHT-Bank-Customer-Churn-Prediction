import streamlit as st

import analytics
import config as cfg
import state
import ui


def hero() -> None:
    st.markdown(
        '<div class="hero rise"><div class="eyebrow">Overview</div>'
        '<h1>FinSIGHT — Customer Churn Intelligence</h1>'
        '<p>FinSIGHT scores bank customers for churn risk with five machine-learning models trained on 10,000 customer '
        'records. Explore how the models perform on held-out customers whose real outcome is known, or enter a new '
        'customer and see how all five models respond. Every number on these pages comes from the saved project models '
        'and evaluation files.</p></div>', unsafe_allow_html=True)
    st.write("")
    a, b, c, _ = st.columns([1.35, 1.35, 1.1, 2.2])
    if a.button("Predict Customer Churn", type="primary", width="stretch", key="home_cta_predict"):
        ui.go_to("views/predict.py")
    if b.button("Explore test records", width="stretch", key="home_cta_explorer"):
        ui.go_to("views/explorer.py")
    if c.button("Compare models", width="stretch", key="home_cta_compare"):
        ui.go_to("views/compare.py")


def churn_explainer() -> None:
    ui.section_title("What is Customer Churn?")
    st.markdown(
        '<div class="card"><p style="margin:0 0 4px 0;line-height:1.7">'
        "Customer churn occurs when a customer stops using a company's services or leaves the company. In banking, "
        "this may mean that a customer closes their account or ends their relationship with the bank. Churn prediction "
        "uses customer information and machine learning to estimate which customers may be at risk of leaving.</p>"
        '<div class="churn-grid">'
        "<div class='item'><b>Spot customers who need attention</b>It helps identify customers who may need attention.</div>"
        "<div class='item'><b>Support data-driven retention</b>It supports data-driven customer retention strategies.</div>"
        "<div class='item'><b>Investigate departure patterns</b>It allows businesses to investigate patterns associated with customer departure.</div>"
        "<div class='item'><b>Make more informed decisions</b>It helps teams make more informed decisions about retention efforts.</div>"
        "</div></div>", unsafe_allow_html=True)
    st.write("")
    ui.callout("<b>Predictions are estimates, not guarantees.</b> A model that flags a customer is saying the profile resembles "
               "customers who left before. It does not mean this customer will leave, and it does not explain why.")


def kpis(service, split, split_error, df, metrics) -> None:
    k1, k2, k3, k4 = st.columns(4)
    if split is not None:
        n, churned = len(split), int(split.y.sum())
        total = f"{len(df):,}" if df is not None else "the full dataset"
        k1.markdown(ui.kpi_card("Evaluation records (held-out test set)", f"{n:,}",
                                f"Never used for training. {n / len(df) * 100:.0f}% of {total} customers." if df is not None
                                else "Never used for training.", "users"), unsafe_allow_html=True)
        k2.markdown(ui.kpi_card("Actual churn rate (test set)", f"{churned / n * 100:.1f}%",
                                f"{churned:,} churned, {n - churned:,} stayed in the test set.", "trend-down", "high"), unsafe_allow_html=True)
    else:
        k1.markdown(ui.kpi_card("Evaluation records (held-out test set)", "Unavailable", split_error or "Test split not found.", "users"), unsafe_allow_html=True)
        k2.markdown(ui.kpi_card("Actual churn rate (test set)", "Unavailable", "Needs the held-out test split.", "trend-down", "high"), unsafe_allow_html=True)
    ready = len(service.available_keys())
    k3.markdown(ui.kpi_card("Trained models available", f"{ready} of {len(cfg.MODEL_ORDER)}",
                            "KNN, Logistic Regression, Decision Tree, Random Forest, SVM." if ready == len(cfg.MODEL_ORDER)
                            else "Some models failed to load. See New Customer Prediction for details.", "layers", "blue"), unsafe_allow_html=True)
    best = ui.best_model_summary(metrics)
    if best:
        k4.markdown(ui.kpi_card("Best model on held-out data", best["name"], best["caption"], "award", "low", small=True), unsafe_allow_html=True)
    else:
        k4.markdown(ui.kpi_card("Best model on held-out data", "Unavailable", "No verified evaluation metrics found.", "award", "low", small=True), unsafe_allow_html=True)


def render() -> None:
    service = ui.get_service()
    df, metrics = ui.get_dataset(), ui.get_metrics()
    split, split_error = ui.get_split()

    hero()
    churn_explainer()
    st.write("")
    kpis(service, split, split_error, df, metrics)

    # --- who is leaving ------------------------------------------------------
    ui.section_title("Who is leaving", "Churned versus retained customers. Counts come straight from the project data.")
    left, right = st.columns([1, 1.5])
    if df is None:
        ui.empty_state("Dataset not available", "Place Churn_Modelling.csv in the data folder to see churn charts.")
    else:
        stats = analytics.overall_stats(df)
        with left:
            with ui.card("home-donut"):
                if split is not None:
                    scope = st.segmented_control("Customers", ["Full dataset", "Test set"], default="Full dataset",
                                                 key="home_scope", label_visibility="collapsed")
                else:
                    scope = "Full dataset"
                if scope == "Test set" and split is not None:
                    churned = int(split.y.sum())
                    shown = {"retained": len(split) - churned, "churned": churned, "churn_rate": churned / len(split)}
                    ui.plot(ui.donut_churn(shown, "Test set: retained vs churned"))
                else:
                    ui.plot(ui.donut_churn(stats, "Full dataset: retained vs churned"))
        with right:
            with ui.card("home-segments"):
                dim = st.selectbox("Break down by", analytics.DIMENSIONS,
                                   format_func=lambda d: analytics.DIMENSION_TITLES[d], key="home_dim")
                table = analytics.segment_chart_data(df, dim)
                ui.plot(ui.segment_bars(table, f"Retained vs churned by {analytics.DIMENSION_TITLES[dim].lower()}"))

    # --- model charts --------------------------------------------------------
    ui.section_title("How the models perform", "Held-out test results. These describe the evaluation set, not any single customer.")
    left, right = st.columns([1.5, 1])
    with left:
        with ui.card("home-metrics"):
            if metrics is not None:
                ui.plot(ui.metric_bars(metrics, "Held-out test performance by model"))
            else:
                ui.empty_state("No evaluation metrics", "Run compare_models.py to generate outputs/comparison/comparison_metrics.csv.")
    with right:
        with ui.card("home-avp"):
            preds = ui.get_predictions()
            if preds:
                ui.plot(ui.actual_vs_predicted_fig(preds))
                st.caption("Each model uses its own alert line, so flagged counts differ. More flagged customers means "
                           "more leavers caught but also more false alarms.")
            else:
                ui.empty_state("Predictions unavailable", "Needs each model's held-out prediction export.", "activity")

    ui.section_title("What the top model relies on", "Feature importance is available for every model; the best model by ROC-AUC is shown here.")
    best = ui.best_model_summary(metrics)
    top_key = next((k for k in cfg.MODEL_ORDER if best and cfg.MODEL_SPECS[k].name == best["name"]), None)
    importance = ui.get_importance(top_key) if top_key and service.is_available(top_key) else None
    with ui.card("home-importance"):
        if importance is not None:
            ui.plot(ui.importance_bars(importance.head(8), f"What {best['name']} relies on"))
            st.caption("Permutation importance on the test set. It shows what the model relies on, not what causes churn.")
        else:
            ui.empty_state("Feature importance unavailable", "Needs the top model and the saved test split.", "activity")

    # --- history -------------------------------------------------------------
    ui.section_title("Recent predictions", "Stored for this session only. New customer details are not kept.")
    history = state.history_frame()
    if history.empty:
        ui.empty_state("No predictions yet", "Run a test record or a new customer and the results will be logged here.", "activity")
    else:
        ui.show_table(history.tail(5).iloc[::-1].fillna("n/a"))


render()
