"""Streamlit smoke/behaviour tests: every page renders in both themes and the two demo flows work."""
import pytest
from streamlit.testing.v1 import AppTest

import config as cfg
import model_service as ms
import state as app_state

PAGES = ["home", "explorer", "predict", "compare", "insights", "retention", "history"]


def page(name, theme_mode="light", **session):
    at = AppTest.from_file(str(cfg.PROJECT_ROOT / "views" / f"{name}.py"), default_timeout=180)
    defaults = {"theme": theme_mode, "history": [], "customer": None, "all_results": None, "customer_ref": None,
                "test_pick": None, "last_source": None}
    for key, value in {**defaults, **session}.items():
        at.session_state[key] = value
    return at


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("name", PAGES)
def test_page_renders_without_errors(name, mode):
    at = page(name, mode).run()
    assert not at.exception, [e.value for e in at.exception]


def test_entry_point_renders_and_theme_defaults_to_light():
    at = AppTest.from_file(str(cfg.PROJECT_ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["theme"] == "light"


def test_theme_switch_changes_styles_and_survives_reruns():
    at = AppTest.from_file(str(cfg.PROJECT_ROOT / "app.py"), default_timeout=180).run()
    css_light = " ".join(m.value for m in at.markdown if "<style>" in m.value)
    assert "--bg:#F6F7FB" in css_light and "color-scheme:light" in css_light
    at.toggle(key="theme_toggle_sidebar").set_value(True).run()
    assert at.session_state["theme"] == "dark"
    css_dark = " ".join(m.value for m in at.markdown if "<style>" in m.value)
    assert "--bg:#0A0F1F" in css_dark and "color-scheme:dark" in css_dark
    at.run()  # an ordinary rerun must not reset the theme
    assert at.session_state["theme"] == "dark"
    assert at.toggle(key="theme_toggle_sidebar").value is True
    at.button(key="theme_button_top").click().run()   # the header button flips it back
    assert at.session_state["theme"] == "light"
    assert at.toggle(key="theme_toggle_sidebar").value is False


# ---------------------------------------------------------------- Test Dataset Explorer
def test_explorer_empty_state_before_any_record():
    at = page("explorer").run()
    assert not at.exception
    assert any("No record selected yet" in m.value for m in at.markdown)


def test_explorer_random_record_is_aligned_and_scored_by_all_models():
    split, _ = ms.load_test_split(), None
    at = page("explorer").run()
    seen = set()
    for _ in range(6):
        at.button(key="explorer_generate").click().run()
        assert not at.exception and not at.error
        pick = at.session_state["test_pick"]
        pos = pick["pos"]
        seen.add(pos)
        assert 0 <= pos < len(split)
        assert pick["actual"] == int(split.y[pos]) and pick["customer_id"] == int(split.customer_ids[pos])
        assert set(pick["results"]) == set(cfg.MODEL_ORDER)
        assert all(isinstance(r, ms.PredictionResult) for r in pick["results"].values())
        text = " ".join(m.value for m in at.markdown)
        n_correct = sum(r.churn == bool(pick["actual"]) for r in pick["results"].values())
        assert f'<div class="score">{n_correct}<small> of 5</small>' in text
        assert text.count("Correct</span>") + text.count("Incorrect</span>") >= 5
    assert len(seen) > 1, "random button kept returning the same record"
    assert len(at.session_state["history"]) == 6 * 5
    assert all(row["Actual"] in ("Stayed", "Churned") for row in at.session_state["history"])


def test_explorer_specific_record_load():
    at = page("explorer").run()
    at.number_input(key="explorer_pos").set_value(1234)
    at.button(key="explorer_load").click().run()
    assert not at.exception
    assert at.session_state["test_pick"]["pos"] == 1234


def test_explorer_reports_missing_split(monkeypatch):
    import ui
    monkeypatch.setattr(ui, "get_split", lambda: (None, "Missing file(s) needed for the held-out test split: test manifest."))
    at = page("explorer").run()
    assert not at.exception
    assert any("Held-out test split unavailable" in m.value for m in at.markdown)
    assert any("Cannot run this mode" in e.value for e in at.error)


# ---------------------------------------------------------------- New Customer Prediction
def test_predict_flow_runs_all_five_real_models_and_does_not_claim_correctness():
    at = page("predict").run()
    at.button[0].click().run()          # the form's submit button
    assert not at.exception and not at.error
    results = at.session_state["all_results"]
    assert set(results) == set(cfg.MODEL_ORDER)
    assert all(isinstance(r, ms.PredictionResult) for r in results.values())
    text = " ".join(m.value for m in at.markdown)
    assert "Unknown outcome" in text
    assert "Correct</span>" not in text and "Incorrect</span>" not in text
    history = at.session_state["history"]
    assert len(history) == 5 and all(row["Actual"] is None for row in history)
    assert all("Age" not in row and "customer" not in row for row in history)
    svm_row = next(r for r in history if r["Model"] == "Support Vector Machine")
    assert svm_row["Churn score"] is None  # no invented probability for the SVM


def test_predict_matches_direct_service_call():
    at = page("predict").run()
    at.button[0].click().run()
    customer = at.session_state["customer"]
    direct = ms.ModelService().predict_all(customer)
    for key in cfg.MODEL_ORDER:
        assert at.session_state["all_results"][key].churn == direct[key].churn
        assert at.session_state["all_results"][key].score == pytest.approx(direct[key].score)


def test_analyze_another_customer_clears_results():
    at = page("predict").run()
    at.button[0].click().run()
    at.button(key="predict_again").click().run()
    assert at.session_state["all_results"] is None and at.session_state["customer"] is None


# ---------------------------------------------------------------- Comparison / history
def test_compare_empty_state_without_customer():
    at = page("compare").run()
    assert not at.exception
    assert any("No customer to compare yet" in m.value for m in at.markdown)


def test_compare_identifies_test_record_vs_new_customer():
    explorer = page("explorer").run()
    explorer.button(key="explorer_generate").click().run()
    cmp_ = page("compare", test_pick=explorer.session_state["test_pick"], last_source=app_state.SOURCE_TEST).run()
    assert not cmp_.exception
    assert any("Held-out test record" in m.value for m in cmp_.markdown)

    predict = page("predict").run()
    predict.button[0].click().run()
    cmp2 = page("compare", all_results=predict.session_state["all_results"], customer=predict.session_state["customer"],
                customer_ref="FS-TEST01", last_source=app_state.SOURCE_NEW).run()
    text = " ".join(m.value for m in cmp2.markdown)
    assert "New customer" in text and "outcome unknown" in text


def test_history_clear_button_works():
    explorer = page("explorer").run()
    explorer.button(key="explorer_generate").click().run()
    hist = page("history", history=explorer.session_state["history"]).run()
    assert not hist.exception
    hist.button(key="history_clear").click().run()
    assert hist.session_state["history"] == []
    assert any("No predictions yet" in m.value for m in hist.markdown)


def test_headline_model_name_maps_to_a_registered_model():
    import ui
    summary = ui.best_model_summary(ms.load_metrics())
    assert summary["name"] in {spec.name for spec in cfg.MODEL_SPECS.values()}
