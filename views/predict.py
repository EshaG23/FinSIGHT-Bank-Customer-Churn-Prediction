import streamlit as st

import analytics
import config as cfg
import model_service as ms
import state
import ui

DEFAULTS = {"CreditScore": 650, "Age": 40, "Tenure": 5, "Balance": 75000.0, "NumOfProducts": 1,
            "EstimatedSalary": 90000.0, "Geography": "France", "Gender": "Female",
            "HasCrCard": True, "IsActiveMember": True}


def form_section(num: int, title: str, desc: str) -> None:
    st.markdown(f'<div class="form-section"><span class="num">{num}</span><div>'
                f'<div class="t">{title}</div><div class="d">{desc}</div></div></div>', unsafe_allow_html=True)


def customer_form(service: ms.ModelService):
    """Returns (submitted, raw_customer). Only the 10 raw fields the models were trained on are collected."""
    with st.form("customer_form", border=False):
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            with ui.card("form-personal"):
                form_section(1, "Personal information", "Who the customer is.")
                age = st.number_input("Age", 18, 100, DEFAULTS["Age"], step=1, help="Whole years, 18 to 100.")
                gender = st.selectbox("Gender", cfg.GENDERS, index=cfg.GENDERS.index(DEFAULTS["Gender"]))
                geography = st.selectbox("Geography", cfg.GEOGRAPHIES, index=cfg.GEOGRAPHIES.index(DEFAULTS["Geography"]),
                                         help="Country of the customer's account.")
        with c2:
            with ui.card("form-financial"):
                form_section(2, "Financial information", "Standing and earning power.")
                credit = st.number_input("Credit score", 300, 900, DEFAULTS["CreditScore"], step=1,
                                         help="Training data covers 350 to 850.")
                balance = st.number_input("Account balance", 0.0, 1_000_000.0, DEFAULTS["Balance"], step=1000.0, format="%.2f",
                                          help="Current balance. Zero is common in the data.")
                salary = st.number_input("Estimated salary", 0.0, 1_000_000.0, DEFAULTS["EstimatedSalary"], step=1000.0,
                                         format="%.2f", help="Estimated yearly salary.")
        with c3:
            with ui.card("form-activity"):
                form_section(3, "Banking activity", "How the customer uses the bank.")
                tenure = st.number_input("Tenure (years)", 0, 10, DEFAULTS["Tenure"], step=1, help="Years with the bank, 0 to 10.")
                products = st.number_input("Number of products", 1, 4, DEFAULTS["NumOfProducts"], step=1,
                                           help="Accounts, cards and loans held, 1 to 4.")
                card = st.toggle("Has credit card", value=DEFAULTS["HasCrCard"])
                active = st.toggle("Is active member", value=DEFAULTS["IsActiveMember"])
        st.caption("Customer ID, surname and row number are not inputs. The models were trained without them.")
        submitted = st.form_submit_button("Predict Customer Churn", type="primary", width="stretch",
                                          disabled=not service.available_keys())
    raw = {"CreditScore": credit, "Age": age, "Tenure": tenure, "Balance": balance, "NumOfProducts": products,
           "HasCrCard": int(card), "IsActiveMember": int(active), "EstimatedSalary": salary,
           "Geography": geography, "Gender": gender}
    return submitted, raw


def run_prediction(service: ms.ModelService, raw: dict) -> bool:
    try:
        clean, warnings_ = ms.validate_customer(raw)
    except ms.ValidationError as exc:
        st.error("Please fix the following before predicting:\n\n" + "\n".join(f"- {p}" for p in exc.problems))
        return False
    for message in warnings_:
        st.warning(message)
    with st.spinner("Scoring customer with all five models"):
        try:
            results = service.predict_all(clean)
        except Exception as exc:  # shown to the user, never replaced by a made-up prediction
            ui.friendly_error("Prediction failed.", exc)
            return False
    ok, _ = ui.split_results(results)
    if not ok:
        st.error("None of the models could score this customer. See the details below.")
    ref = state.new_reference()
    state.record_run(ref, state.SOURCE_NEW, results)
    st.session_state.update(customer=clean, all_results=results, customer_ref=ref, last_source=state.SOURCE_NEW)
    return True


def what_if(service: ms.ModelService, customer: dict, results: dict) -> None:
    variants = analytics.what_if_variants(customer)
    if not variants:
        return
    ok, _ = ui.split_results(results)
    base = sum(r.churn for r in ok.values())
    ui.section_title("Model sensitivity", "How many models flag the customer if one field changes.")
    shown = False
    for label, changed in variants:
        try:
            other, _ = ui.split_results(service.predict_all(changed))
        except Exception:
            continue
        if not other:
            continue
        shown = True
        now = sum(r.churn for r in other.values())
        ui.callout(f"<b>{label}:</b> {now} of {len(other)} models flag churn (currently {base} of {len(ok)}).")
        st.write("")
    if shown:
        st.caption("This probes the models only. It does not predict what an intervention would actually achieve.")


def render_results(service: ms.ModelService) -> None:
    results = st.session_state["all_results"]
    customer = st.session_state["customer"]
    ok, failed = ui.split_results(results)
    flagged = sum(r.churn for r in ok.values())

    ui.section_title("Prediction results",
                     "Outcome unknown. These are model estimates for a customer who is not in the test set, so they cannot be marked correct or incorrect.")
    if ok:
        with ui.card("predict-summary"):
            st.markdown(
                f'<div class="verdict rise"><div><div class="lbl">Customer</div>'
                f'<div class="big">{ui.esc(st.session_state.get("customer_ref") or "New customer")}</div></div>'
                f'<div><div class="lbl">Ground truth</div>{ui.badge("Unknown outcome", "neutral", icon_name="help")}</div>'
                f'<div><div class="lbl">Models predicting churn</div>'
                f'<div class="score">{flagged}<small> of {len(ok)}</small></div></div></div>', unsafe_allow_html=True)
        st.write("")
    ui.model_cards(results)
    st.write("")
    st.markdown(ui.prediction_table(results), unsafe_allow_html=True)
    st.caption("Churn probability is each model's raw 0 to 100% output, not a calibrated probability, and no model is certain. "
               "The SVM only has a decision margin, so no probability is shown for it.")
    st.write("")
    html_text, kind = ui.agreement_html(results)
    ui.callout(html_text, kind)
    for key, message in failed.items():
        st.warning(f"{cfg.MODEL_SPECS[key].name} is unavailable: {message}")

    rows = [{"key": r.model_key, "name": r.model_name, "pct": r.percentile, "thr": r.threshold_percentile, "churn": r.churn}
            for r in ok.values() if r.percentile is not None and r.threshold_percentile is not None]
    if rows:
        with ui.card("predict-rank"):
            ui.plot(ui.rank_chart(rows))
        st.caption("Ranking puts all five models on one scale. A bar that passes the marker means that model flags the customer.")

    rates = ui.get_rates()
    if rates is not None:
        signals = analytics.customer_signals(customer, rates)
        if signals:
            ui.section_title("Profile signals from the data", "Historical churn rate of groups this customer belongs to.")
            rows_html = ""
            for s in signals:
                cls, sign = ("up", "+") if s["Direction"] == "higher" else ("down", "")
                rows_html += (f'<div class="signal"><span class="pill {cls}">{sign}{s["GapPoints"]:.0f} pts</span>'
                              f'<span><b>{ui.esc(analytics.DIMENSION_TITLES[s["Dimension"]])}: {ui.esc(s["Segment"])}</b> churned at '
                              f'{s["ChurnRate"] * 100:.1f}% across {s["Customers"]:,} customers.</span></div>')
            st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)
            st.caption("Points are versus the overall churn rate in the dataset. These are associations, not causes.")
    if ok:
        what_if(service, customer, results)

    st.write("")
    b1, b2, b3 = st.columns(3)
    if b1.button("Analyze another customer", type="primary", width="stretch", key="predict_again"):
        state.reset_customer()
        st.rerun()
    if b2.button("Compare all five predictions", width="stretch", key="predict_to_compare"):
        ui.go_to("views/compare.py")
    if b3.button("View model performance", width="stretch", key="predict_to_insights"):
        ui.go_to("views/insights.py")


def render() -> None:
    service = ui.get_service()
    ui.page_header("Demonstration mode 2", "New Customer Prediction — Interactive Churn Analysis",
                   "In this mode, you can enter customer details and see how the five trained models predict the customer's churn "
                   "outcome. Because this customer may not have a known outcome, the predictions cannot automatically be "
                   "classified as correct or incorrect.")

    for status in (s for s in service.status.values() if not s.available):
        st.error(f"{status.name} is unavailable. {status.error}")
    if not service.available_keys():
        ui.empty_state("No models could be loaded", "Check that the outputs folder contains the saved model files. See the README.", "alert")
        return

    submitted, raw = customer_form(service)
    if submitted:
        run_prediction(service, raw)

    if st.session_state.get("all_results") is not None:
        render_results(service)
    else:
        st.write("")
        ui.empty_state("Your results will appear here",
                       "Fill in the customer profile, then press Predict Customer Churn to see all five models side by side.", "activity")


render()
