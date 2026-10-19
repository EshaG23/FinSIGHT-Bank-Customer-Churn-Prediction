"""Tests for the FinSIGHT prediction pipeline.

The key idea: raw customer fields (strings like "Germany") go through the same
validate -> encode -> scale -> score path a new customer would use, and the
results must reproduce what the training scripts saved for the held-out test
customers. Passing only on pre-encoded test rows would not prove that.
"""
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

import analytics
import config as cfg
import model_service as ms

NEW_CUSTOMER = {"CreditScore": 640, "Age": 47, "Tenure": 3, "Balance": 118000.0, "NumOfProducts": 1,
                "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 72000.0,
                "Geography": "Germany", "Gender": "Female"}


@pytest.fixture(scope="module")
def service():
    return ms.ModelService()


@pytest.fixture(scope="module")
def raw_test_rows():
    raw = pd.read_csv(cfg.DATA_FILE)
    manifest = pd.read_csv(cfg.OUTPUTS_DIR / "comparison" / "canonical_test_manifest.csv")
    rows = raw.iloc[manifest.source_row_position.to_numpy()].reset_index(drop=True)
    assert (rows.CustomerId.to_numpy() == manifest.CustomerId.to_numpy()).all()
    return rows


def copy_outputs(tmp_path):
    target = tmp_path / "outputs"
    shutil.copytree(cfg.OUTPUTS_DIR, target)
    return target


# --- loading ---------------------------------------------------------------
def test_all_five_models_load(service):
    assert service.available_keys() == cfg.MODEL_ORDER
    for status in service.status.values():
        assert status.available and status.error is None and status.threshold is not None


# --- schema and preprocessing artifacts -----------------------------------
def test_feature_schema_matches_every_saved_artifact(service):
    saved = joblib.load(cfg.OUTPUTS_DIR / cfg.GLOBAL_FEATURE_COLUMNS)
    assert saved == cfg.FEATURE_COLUMNS
    assert joblib.load(cfg.OUTPUTS_DIR / "scale_columns.pkl") == cfg.SCALE_COLUMNS
    for key, model in service.models.items():
        assert model.feature_columns == cfg.FEATURE_COLUMNS, key


def test_scaler_statistics_agree_across_artifacts(service):
    global_scaler = joblib.load(cfg.OUTPUTS_DIR / cfg.GLOBAL_SCALER)
    svm = json.loads((cfg.OUTPUTS_DIR / cfg.MODEL_SPECS["svm"].artifact).read_text())
    numeric_idx = [cfg.FEATURE_COLUMNS.index(c) for c in cfg.SCALE_COLUMNS]
    reference = {"mean": global_scaler.mean_, "scale": global_scaler.scale_}
    np.testing.assert_allclose(service.models["logistic"].mean, reference["mean"], rtol=1e-9)
    np.testing.assert_allclose(service.models["logistic"].scale, reference["scale"], rtol=1e-9)
    np.testing.assert_allclose(np.array(svm["scaler_mean"])[numeric_idx], reference["mean"], rtol=1e-9)
    np.testing.assert_allclose(np.array(svm["scaler_scale"])[numeric_idx], reference["scale"], rtol=1e-9)
    np.testing.assert_allclose(service.models["knn"].mean, reference["mean"], rtol=1e-9)
    np.testing.assert_allclose(service.models["knn"].scale, reference["scale"], rtol=1e-9)


def test_encoding_matches_saved_test_features(raw_test_rows):
    encoded = ms.encode_customers(raw_test_rows)
    saved = pd.read_csv(cfg.OUTPUTS_DIR / cfg.X_TEST_ENCODED)
    assert list(encoded.columns) == cfg.FEATURE_COLUMNS == list(saved.columns)
    assert encoded.shape == saved.shape == (2000, 13)
    np.testing.assert_allclose(encoded.to_numpy(), saved.to_numpy())


def test_single_customer_encoding_shape_and_order():
    clean, _ = ms.validate_customer(NEW_CUSTOMER)
    encoded = ms.encode_customer(clean)
    assert encoded.shape == (1, 13)
    assert list(encoded.columns) == cfg.FEATURE_COLUMNS
    row = encoded.iloc[0]
    assert (row.Geography_Germany, row.Geography_France, row.Geography_Spain) == (1.0, 0.0, 0.0)
    assert (row.Gender_Female, row.Gender_Male) == (1.0, 0.0)


# --- the real requirement: new-customer inference reproduces saved results ---
@pytest.mark.parametrize("key", cfg.MODEL_ORDER)
def test_raw_input_path_reproduces_saved_test_predictions(service, raw_test_rows, key):
    saved = ms.load_predictions(key)
    results = service.predict_encoded(key, ms.encode_customers(raw_test_rows))
    scores = np.array([r.score for r in results])
    flags = np.array([r.churn for r in results]).astype(int)
    np.testing.assert_allclose(scores, saved.y_score.to_numpy(), atol=1e-9)
    assert (flags == saved.y_pred.to_numpy()).all()


@pytest.mark.parametrize("key", cfg.MODEL_ORDER)
def test_single_customer_api_matches_batch(service, raw_test_rows, key):
    saved = ms.load_predictions(key)
    for i in np.random.default_rng(7).choice(2000, 25, replace=False):
        customer = {c: raw_test_rows.loc[i, c].item() if hasattr(raw_test_rows.loc[i, c], "item") else raw_test_rows.loc[i, c]
                    for c in ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
                              "IsActiveMember", "EstimatedSalary", "Geography", "Gender"]}
        result = service.predict(key, customer)
        assert result.score == pytest.approx(saved.y_score.iloc[i], abs=1e-9)
        assert int(result.churn) == int(saved.y_pred.iloc[i])


def test_saved_predictions_are_consistent_with_thresholds(service):
    for key in cfg.MODEL_ORDER:
        saved = ms.load_predictions(key)
        assert ((saved.y_score >= service.models[key].threshold).astype(int) == saved.y_pred).all(), key


# --- synthetic customers -----------------------------------------------------
def test_synthetic_customer_returns_valid_result_from_all_models(service):
    results = service.predict_all(NEW_CUSTOMER)
    assert list(results) == cfg.MODEL_ORDER
    for key, result in results.items():
        assert isinstance(result, ms.PredictionResult), f"{key}: {result}"
        assert np.isfinite(result.score)
        assert result.label in ("Likely to Churn", "Likely to Stay")
        assert result.risk_tier in ("Low", "Moderate", "High")
        assert 0 <= result.percentile <= 100


def test_probability_availability_is_per_model(service):
    results = service.predict_all(NEW_CUSTOMER)
    for key in ("knn", "logistic", "tree", "forest"):
        assert results[key].has_probability_scale
        assert 0.0 <= results[key].score <= 1.0
    assert not results["svm"].has_probability_scale   # margin only, never presented as a probability


def test_models_rank_risky_profile_above_safe_profile(service):
    risky = {**NEW_CUSTOMER, "Age": 54, "Geography": "Germany", "IsActiveMember": 0, "NumOfProducts": 1}
    safe = {**NEW_CUSTOMER, "Age": 28, "Geography": "France", "IsActiveMember": 1, "NumOfProducts": 2, "Balance": 0.0}
    for key in cfg.MODEL_ORDER:
        assert service.predict(key, risky).score > service.predict(key, safe).score, key


def test_risk_tier_rules(service):
    for key in cfg.MODEL_ORDER:
        r = service.predict(key, NEW_CUSTOMER)
        assert (r.risk_tier == "High") == r.churn


# --- invalid input -----------------------------------------------------------
@pytest.mark.parametrize("field,value", [
    ("Age", 5), ("Age", 150), ("CreditScore", 1200), ("Balance", -1.0), ("NumOfProducts", 9),
    ("Tenure", 11), ("EstimatedSalary", -5.0), ("HasCrCard", 2), ("Geography", "Atlantis"), ("Gender", "Other"),
    ("Age", "abc"), ("Balance", float("nan")), ("Age", None), ("Age", 40.5), ("Geography", None)])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ms.ValidationError) as caught:
        ms.validate_customer({**NEW_CUSTOMER, field: value})
    assert caught.value.problems


def test_missing_fields_are_all_reported():
    with pytest.raises(ms.ValidationError) as caught:
        ms.validate_customer({"Age": 40})
    assert len(caught.value.problems) >= 8


def test_invalid_input_never_produces_a_prediction(service):
    with pytest.raises(ms.ValidationError):
        service.predict("forest", {**NEW_CUSTOMER, "Age": -3})
    with pytest.raises(ms.ValidationError):
        service.predict_all({**NEW_CUSTOMER, "Geography": "Mars"})


def test_unknown_category_in_batch_is_rejected_not_zero_filled():
    frame = pd.DataFrame([{**NEW_CUSTOMER, "Geography": "Mars"}])
    with pytest.raises(ms.ValidationError):
        ms.encode_customers(frame)


def test_out_of_training_range_gives_warning_not_error():
    clean, warnings_ = ms.validate_customer({**NEW_CUSTOMER, "Age": 96, "Balance": 400000.0})
    assert clean["Age"] == 96
    assert len(warnings_) == 2


# --- missing / broken artifacts --------------------------------------------
@pytest.mark.parametrize("key", ["logistic", "tree", "forest", "svm"])
def test_missing_model_file_marks_only_that_model_unavailable(tmp_path, key):
    outputs = copy_outputs(tmp_path)
    (outputs / cfg.MODEL_SPECS[key].artifact).unlink()
    svc = ms.ModelService(outputs)
    assert not svc.status[key].available and "Missing model file" in svc.status[key].error
    assert sorted(svc.available_keys()) == sorted(set(cfg.MODEL_ORDER) - {key})
    with pytest.raises(ms.ModelUnavailableError):
        svc.predict(key, NEW_CUSTOMER)
    results = svc.predict_all(NEW_CUSTOMER)
    assert isinstance(results[key], str)                       # error text, not a fake prediction
    assert all(isinstance(results[k], ms.PredictionResult) for k in cfg.MODEL_ORDER if k != key)


def test_missing_knn_training_data_marks_knn_unavailable(tmp_path):
    outputs = copy_outputs(tmp_path)
    (outputs / cfg.X_TRAIN_ENCODED).unlink()
    svc = ms.ModelService(outputs)
    assert not svc.status["knn"].available
    assert all(svc.status[k].available for k in ("logistic", "tree", "forest", "svm"))


def test_missing_knn_settings_marks_knn_unavailable(tmp_path):
    outputs = copy_outputs(tmp_path)
    (outputs / cfg.KNN_SETTINGS).unlink()
    assert not ms.ModelService(outputs).status["knn"].available


def test_corrupt_artifact_is_reported_not_swallowed(tmp_path):
    outputs = copy_outputs(tmp_path)
    (outputs / cfg.MODEL_SPECS["svm"].artifact).write_text("{ not json")
    (outputs / cfg.MODEL_SPECS["tree"].artifact).write_bytes(b"garbage")
    svc = ms.ModelService(outputs)
    assert not svc.status["svm"].available and svc.status["svm"].error
    assert not svc.status["tree"].available and svc.status["tree"].error
    assert svc.status["forest"].available


def test_schema_mismatch_in_artifact_is_rejected(tmp_path):
    outputs = copy_outputs(tmp_path)
    path = outputs / cfg.MODEL_SPECS["svm"].artifact
    params = json.loads(path.read_text())
    params["feature_names"] = list(reversed(params["feature_names"]))
    path.write_text(json.dumps(params))
    svc = ms.ModelService(outputs)
    assert not svc.status["svm"].available and "feature order" in svc.status["svm"].error


def test_all_models_missing_leaves_service_usable(tmp_path):
    empty = tmp_path / "outputs"
    empty.mkdir()
    svc = ms.ModelService(empty)
    assert svc.available_keys() == []
    assert all(isinstance(v, str) for v in svc.predict_all(NEW_CUSTOMER).values())


# --- evaluation data --------------------------------------------------------
def test_published_metrics_match_recomputed_metrics():
    published = ms.load_metrics().set_index("Model")
    for key in cfg.MODEL_ORDER:
        recomputed = ms.metrics_from_predictions(ms.load_predictions(key))
        row = published.loc[cfg.MODEL_SPECS[key].name]
        for metric, value in recomputed.items():
            assert row[metric] == pytest.approx(value, abs=1e-9), (key, metric)


def test_metrics_fall_back_to_saved_predictions_when_summary_missing(tmp_path):
    outputs = copy_outputs(tmp_path)
    (outputs / cfg.COMPARISON_METRICS).unlink()
    fallback = ms.load_metrics(outputs)
    assert sorted(fallback.Model) == sorted(s.name for s in cfg.MODEL_SPECS.values())
    assert {"TN", "FP", "FN", "TP"}.issubset(fallback.columns)


def test_permutation_importance_runs_for_every_model(service):
    for key in cfg.MODEL_ORDER:
        table = ms.permutation_importance(service, key, repeats=1)
        assert set(table.Feature) == {"CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
                                      "IsActiveMember", "EstimatedSalary", "Geography", "Gender"}
        assert np.isfinite(table.Importance).all()


# --- dataset analytics ------------------------------------------------------
def test_dataset_statistics_and_segments():
    df = ms.load_dataset()
    stats = analytics.overall_stats(df)
    assert stats["customers"] == 10000 and stats["churned"] == 2037
    rates = analytics.segment_rates(df)
    germany = rates[(rates.Dimension == "Geography") & (rates.Segment == "Germany")].iloc[0]
    assert (germany.Customers, germany.Churned) == (2509, 814)
    for dim in analytics.DIMENSIONS:                       # each dimension covers every customer once
        assert rates[rates.Dimension == dim].Customers.sum() == 10000


def test_customer_signals_point_the_right_way():
    rates = analytics.segment_rates(ms.load_dataset())
    signals = analytics.customer_signals({**NEW_CUSTOMER}, rates)
    assert signals and all(s["Direction"] == "higher" for s in signals if s["Dimension"] in ("Geography", "IsActiveMember"))
