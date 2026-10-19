import numpy as np
import streamlit as st

import config as cfg
import model_service as ms
import state
import ui


def run_record(service: ms.ModelService, split: ms.TestSplit, pos: int) -> None:
    """Score one held-out record with every model. Row `pos` of features and labels is the same customer."""
    customer_id, actual = int(split.customer_ids[pos]), int(split.y[pos])
    with st.spinner("Scoring record with all five models"):
        results = service.predict_all_encoded(split.encoded.iloc[[pos]])
    state.record_run(f"Test #{pos} (ID {customer_id})", state.SOURCE_TEST, results, actual)
    st.session_state.update(test_pick={"pos": pos, "customer_id": customer_id, "actual": actual, "results": results},
                            last_source=state.SOURCE_TEST)


def random_position(split: ms.TestSplit) -> int:
    previous = (st.session_state.get("test_pick") or {}).get("pos")
    rng = np.random.default_rng()            # fresh entropy each click: a different record every time
    pos = int(rng.integers(len(split)))
    while len(split) > 1 and pos == previous:
        pos = int(rng.integers(len(split)))
    return pos


def integrity_note(pick: dict) -> None:
    """Confirm live predictions equal what the evaluation scripts exported for this customer."""
    checked, mismatched = 0, []
    for key, result in pick["results"].items():
        if not isinstance(result, ms.PredictionResult):
            continue
        saved = ms.saved_prediction(key, pick["customer_id"])
        if saved is None:
            continue
        checked += 1
        if saved["y_pred"] != int(result.churn) or abs(saved["y_score"] - result.score) > 1e-6:
            mismatched.append(result.model_name)
    if mismatched:
        st.warning("Live predictions differ from the saved evaluation export for: " + ", ".join(mismatched)
                   + ". Re-run regenerate_predictions.py or check the model files.")
    elif checked:
        st.caption(f"Integrity check: live predictions match the saved evaluation export for {checked} of "
                   f"{len(cfg.MODEL_ORDER)} models on this customer.")


def render_pick(split: ms.TestSplit) -> None:
    pick = st.session_state["test_pick"]
    pos, actual, results = pick["pos"], pick["actual"], pick["results"]
    ok, failed = ui.split_results(results)
    correct = sum(r.churn == bool(actual) for r in ok.values())

    with ui.card("explorer-verdict"):
        st.markdown(
            f'<div class="verdict rise"><div><div class="lbl">Held-out test record</div>'
            f'<div class="big">#{pos:,} <span class="muted-text" style="font-size:1rem;font-weight:500">of {len(split):,} · Customer ID {pick["customer_id"]}</span></div></div>'
            f'<div><div class="lbl">Actual outcome (ground truth)</div>{ui.actual_badge(actual)}</div>'
            f'<div><div class="lbl">Models predicting this record correctly</div>'
            f'<div class="score">{correct}<small> of {len(ok)}</small></div></div></div>', unsafe_allow_html=True)

    ui.section_title("Customer feature summary", "The record's own values from the held-out test split. Nothing here is generated.")
    with ui.card("explorer-fields"):
        st.markdown(ui.fields_html(ui.customer_fields(split.raw.iloc[pos].to_dict())), unsafe_allow_html=True)

    ui.section_title("Model predictions vs actual outcome",
                     "Each model applies its own preprocessing and alert line to this same record.")
    st.markdown(ui.prediction_table(results, actual), unsafe_allow_html=True)
    st.caption("Churn probability is each model's raw 0 to 100% output, not a calibrated probability. The SVM only produces a "
               "decision margin, so no probability is shown for it. A model predicts churn when its score reaches its own alert line.")
    st.write("")
    html_text, kind = ui.agreement_html(results, actual)
    ui.callout(html_text, kind)
    for key, message in failed.items():
        st.warning(f"{cfg.MODEL_SPECS[key].name} is unavailable: {message}")

    rows = [{"key": r.model_key, "name": r.model_name, "pct": r.percentile, "thr": r.threshold_percentile, "churn": r.churn}
            for r in ok.values() if r.percentile is not None and r.threshold_percentile is not None]
    if rows:
        with ui.card("explorer-rank"):
            ui.plot(ui.rank_chart(rows))
        st.caption("Ranking puts all five models on one scale. A bar that passes the marker means that model flags the customer.")
    integrity_note(pick)

    ui.callout("<b>One record is an illustration, not an accuracy score.</b> Whether a model gets this single customer right says "
               "little about how good it is overall. Overall accuracy, precision, recall and ROC-AUC are measured on all 2,000 "
               "test records on the Model Comparison page.")
    st.write("")
    a, b, _ = st.columns([1.4, 1.4, 2.2])
    if a.button("Generate another random record", type="primary", width="stretch", key="explorer_again"):
        st.session_state["_explorer_pending"] = "random"
        st.rerun()
    if b.button("Open detailed comparison", width="stretch", key="explorer_to_compare"):
        ui.go_to("views/compare.py")


def render() -> None:
    service = ui.get_service()
    ui.page_header("Demonstration mode 1", "Test Dataset Explorer — Evaluate on Known Customers",
                   "This mode selects a customer from the held-out test dataset. Since the customer's actual churn label is known, "
                   "we can compare the predictions of all five models with the ground-truth outcome.")

    split, error = ui.get_split()
    if split is None:
        ui.empty_state("Held-out test split unavailable", error or "The saved test split could not be loaded.", "alert")
        st.error(f"Cannot run this mode: {error}")
        return
    if not service.available_keys():
        ui.empty_state("No models could be loaded", "Check that the outputs folder contains the saved model files. See the README.", "alert")
        return
    for status in (s for s in service.status.values() if not s.available):
        st.error(f"{status.name} is unavailable. {status.error}")

    st.markdown(f'<div class="section-sub" style="margin-top:-6px">Test split: <b>{len(split):,}</b> customers '
                f'({split.y.mean() * 100:.1f}% churned), separate from the training data. Nothing is fitted or changed in this mode.</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 3])
    clicked = c1.button("Generate Random Test Record", type="primary", width="stretch", key="explorer_generate")
    with c2.expander("Inspect a specific record"):
        e1, e2 = st.columns([2, 1])
        wanted = e1.number_input("Record number", 0, len(split) - 1, 0, step=1, key="explorer_pos",
                                 help=f"Position in the test split, 0 to {len(split) - 1}.")
        load = e2.button("Load record", key="explorer_load", width="stretch")

    pending = st.session_state.pop("_explorer_pending", None)
    if clicked or pending == "random":
        run_record(service, split, random_position(split))
    elif load:
        run_record(service, split, int(wanted))

    if st.session_state.get("test_pick") is None:
        st.write("")
        ui.empty_state("No record selected yet", "Press Generate Random Test Record to draw a customer from the held-out test set "
                       "and see how all five models do against the known outcome.", "target")
    else:
        render_pick(split)


render()
