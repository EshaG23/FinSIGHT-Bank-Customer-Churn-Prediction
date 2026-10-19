"""Theme tokens (contrast), themed charts/tables, held-out split alignment and failure handling."""
import shutil

import numpy as np
import pandas as pd
import pytest

import config as cfg
import model_service as ms
import theme
import ui


# ------------------------------------------------------------------ design tokens
@pytest.mark.parametrize("mode", theme.MODES)
def test_text_pairs_meet_wcag_aa(mode):
    p = theme.PALETTES[mode]
    pairs = [("text", "bg"), ("text", "surface"), ("text", "surface_2"), ("muted", "bg"), ("muted", "surface"),
             ("muted", "surface_2"), ("subtle", "surface"), ("accent_text", "surface"), ("accent_fg", "accent"),
             ("accent_fg", "accent_hover")]
    if mode == theme.LIGHT:   # solid status fills in light mode
        pairs += [("good_fg", "good_bg"), ("bad_fg", "bad_bg"), ("warn_fg", "warn_bg")]
    for fg, bg in pairs:
        ratio = theme.contrast_ratio(p[fg], p[bg])
        assert ratio >= 4.5, f"{mode}: {fg} on {bg} is only {ratio:.2f}:1"
    if mode == theme.DARK:    # status text sits on a translucent tint over the surface
        for fg in ("good_fg", "bad_fg", "warn_fg"):
            assert theme.contrast_ratio(p[fg], p["surface"]) >= 4.5


@pytest.mark.parametrize("mode", theme.MODES)
def test_chart_labels_readable_on_marks(mode):
    p = theme.PALETTES[mode]
    label = "#FFFFFF" if mode == theme.LIGHT else "#0A0F1F"   # colours used for text drawn on bars/slices
    marks = list(theme.MODEL_COLORS[mode].values()) + [p["retained"], p["churned"]]
    for colour in marks:
        assert theme.contrast_ratio(label, colour) >= 4.5, (mode, colour)
    for key in ("text", "muted"):
        assert theme.contrast_ratio(p[key], p["surface"]) >= 4.5


def test_light_and_dark_palettes_define_identical_keys_and_differ():
    assert theme.PALETTES["light"].keys() == theme.PALETTES["dark"].keys()
    assert theme.PALETTES["light"]["bg"] != theme.PALETTES["dark"]["bg"]
    for mode in theme.MODES:
        css = theme.build_css(mode)
        assert css.startswith(":root{") and f"color-scheme:{mode}" in css
        assert "--text:" in css and "--surface:" in css


def test_stylesheet_uses_variables_not_hardcoded_text_colours():
    css = (cfg.ASSETS_DIR / "styles.css").read_text()
    # colours must come from variables; only white on accent/knobs and shadows may be literal
    literals = {c.upper() for c in __import__("re").findall(r"#[0-9a-fA-F]{6}\b", css)}
    assert literals <= {"#FFFFFF"}, literals


# ------------------------------------------------------------------ charts follow the theme
@pytest.mark.parametrize("mode", theme.MODES)
def test_figures_use_the_active_theme(mode, monkeypatch):
    monkeypatch.setattr(theme, "current", lambda: mode)
    p = theme.PALETTES[mode]
    fig = ui.style_fig(ui.roc_fig())
    assert fig.layout.font.color == p["text"]
    assert fig.layout.xaxis.tickfont.color == p["muted"] and fig.layout.xaxis.gridcolor == p["plot_grid"]
    assert fig.layout.hoverlabel.bgcolor == p["hover_bg"] and fig.layout.hoverlabel.font.color == p["text"]
    assert fig.layout.legend.font.color == p["muted"]
    colours = {t.name.split(" (")[0]: t.line.color for t in fig.data if t.name and "AUC" in t.name}
    assert colours["K-Nearest Neighbors"] == theme.MODEL_COLORS[mode]["knn"]   # model identity is stable
    metrics = ms.load_metrics()
    cm = ui.confusion_fig(metrics.iloc[0], "x")
    assert cm.layout.title.font.color == p["text"]


def test_model_keeps_same_colour_across_charts(monkeypatch):
    monkeypatch.setattr(theme, "current", lambda: "dark")
    bars = ui.metric_bars(ms.load_metrics())
    forest = next(t for t in bars.data if t.name == "Random Forest")
    roc = ui.roc_fig()
    forest_roc = next(t for t in roc.data if t.name and t.name.startswith("Random Forest"))
    assert forest.marker.color == forest_roc.line.color == theme.MODEL_COLORS["dark"]["forest"]


def test_html_table_escapes_untrusted_text_and_keeps_trusted_html():
    df = pd.DataFrame({"A": ["<script>x</script>"], "B": ["<b>ok</b>"], "C": [0.5]})
    out = ui.html_table(df, html_cols=["B"], num_cols=["C"], best={"C": "max"})
    assert "&lt;script&gt;" in out and "<script>" not in out and "<b>ok</b>" in out and 'class="num best"' in out


# ------------------------------------------------------------------ held-out split
@pytest.fixture(scope="module")
def split():
    return ms.load_test_split()


@pytest.fixture(scope="module")
def service():
    return ms.ModelService()


def test_split_size_matches_80_20_and_labels_are_binary(split):
    raw_n = len(pd.read_csv(cfg.DATA_FILE))
    assert (raw_n, len(split)) == (10000, 2000) and raw_n - len(split) == len(pd.read_csv(cfg.OUTPUTS_DIR / cfg.Y_TRAIN)) == 8000
    assert set(np.unique(split.y)) == {0, 1} and ms.TARGET_LABELS == {0: "Stayed", 1: "Churned"}


def test_test_and_train_customers_do_not_overlap(split):
    raw = pd.read_csv(cfg.DATA_FILE)
    train_ids = set(raw.CustomerId) - set(split.customer_ids)
    assert len(train_ids) == 8000 and not train_ids & set(split.customer_ids)


def test_row_alignment_features_labels_identities(split):
    raw = pd.read_csv(cfg.DATA_FILE).set_index("CustomerId")
    rng = np.random.default_rng(7)
    for pos in rng.choice(len(split), 60, replace=False):
        cid = int(split.customer_ids[pos])
        assert int(raw.loc[cid, "Exited"]) == int(split.y[pos])
        assert raw.loc[cid, "Age"] == split.raw.iloc[pos]["Age"] == split.encoded.iloc[pos]["Age"]
        assert split.encoded.iloc[pos][f"Geography_{raw.loc[cid, 'Geography']}"] == 1.0


def test_live_scores_for_test_records_reproduce_saved_exports(split, service):
    rng = np.random.default_rng(11)
    for pos in rng.choice(len(split), 40, replace=False):
        results = service.predict_all_encoded(split.encoded.iloc[[pos]])
        for key, result in results.items():
            saved = ms.saved_prediction(key, int(split.customer_ids[pos]))
            assert saved["y_pred"] == int(result.churn), (key, pos)
            assert saved["y_score"] == pytest.approx(result.score, abs=1e-9)


def test_correctness_flag_matches_confusion_counts(split, service):
    """Scoring every test record one at a time must reproduce each model's saved confusion matrix."""
    metrics = ms.load_metrics().set_index("Model")
    rows = range(0, len(split), 10)    # every 10th record is enough to catch an alignment bug cheaply
    for key in cfg.MODEL_ORDER:
        enc = split.encoded.iloc[list(rows)]
        pred = np.array([r.churn for r in service.predict_encoded(key, enc)], dtype=int)
        actual = split.y[list(rows)]
        full = ms.load_predictions(key)
        assert (pred == full.y_pred.to_numpy()[list(rows)]).all()
        assert (actual == full.y_true.to_numpy()[list(rows)]).all()


# ------------------------------------------------------------------ honest failure handling
def _copy_outputs(tmp_path):
    target = tmp_path / "outputs"
    shutil.copytree(cfg.OUTPUTS_DIR, target)
    return target


def test_missing_manifest_gives_clear_error(tmp_path):
    out = _copy_outputs(tmp_path)
    (out / "comparison" / "canonical_test_manifest.csv").unlink()
    with pytest.raises(ms.ArtifactError, match="Missing file"):
        ms.load_test_split(out)


def test_misaligned_labels_are_rejected(tmp_path):
    out = _copy_outputs(tmp_path)
    y = pd.read_csv(out / cfg.Y_TEST)
    y.iloc[:, 0] = y.iloc[:, 0].iloc[::-1].to_numpy()          # same labels, wrong order
    y.to_csv(out / cfg.Y_TEST, index=False)
    with pytest.raises(ms.ArtifactError, match="not aligned"):
        ms.load_test_split(out)


def test_unexpected_label_values_are_rejected(tmp_path):
    out = _copy_outputs(tmp_path)
    y = pd.read_csv(out / cfg.Y_TEST)
    y.iloc[0, 0] = 2
    y.to_csv(out / cfg.Y_TEST, index=False)
    with pytest.raises(ms.ArtifactError):
        ms.load_test_split(out)


def test_one_missing_model_does_not_fake_results(tmp_path, split):
    out = _copy_outputs(tmp_path)
    (out / cfg.MODEL_SPECS["forest"].artifact).unlink()
    svc = ms.ModelService(out)
    assert svc.available_keys() == [k for k in cfg.MODEL_ORDER if k != "forest"]
    results = svc.predict_all_encoded(split.encoded.iloc[[0]])
    assert isinstance(results["forest"], str) and "unavailable" in results["forest"].lower()
    assert all(isinstance(results[k], ms.PredictionResult) for k in cfg.MODEL_ORDER if k != "forest")
    html, kind = ui.agreement_html(results)
    assert "only 4 of 5" in html


def test_invalid_customer_inputs_are_rejected_with_all_problems():
    bad = {"CreditScore": 100, "Age": 17, "Tenure": 11, "Balance": -5, "NumOfProducts": 9, "HasCrCard": 2,
           "IsActiveMember": 1, "EstimatedSalary": 10, "Geography": "Mars", "Gender": "Female"}
    with pytest.raises(ms.ValidationError) as exc:
        ms.validate_customer(bad)
    assert len(exc.value.problems) >= 6


def test_agreement_text_states_known_outcome(split, service):
    results = service.predict_all_encoded(split.encoded.iloc[[0]])
    text, _ = ui.agreement_html(results, actual=int(split.y[0]))
    assert "known outcome" in text or "matches the known outcome" in text
    text_unknown, _ = ui.agreement_html(results, actual=None)
    assert "known outcome" not in text_unknown


def test_pr_curve_points_exist_for_all_models():
    for key in cfg.MODEL_ORDER:
        recall, precision, ap = ms.pr_points(key)
        assert 0 < ap <= 1 and len(recall) == len(precision)
