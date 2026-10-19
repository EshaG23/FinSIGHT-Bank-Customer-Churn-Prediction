"""Model loading, preprocessing and inference for FinSIGHT.

Design notes
------------
* The five models were written from scratch in NumPy (see src/). Only four
  have a saved model artifact. KNN stores its training customers, so it is
  rebuilt here from the saved training split with the validation-selected
  settings (same procedure as regenerate_predictions.py).
* Raw customer fields are one-hot encoded exactly like src/FULL_EDA.py and
  reordered to the saved feature order. Scaling uses statistics saved at
  training time and is never refit on new input.
* Every model keeps its own decision threshold. Scores are NOT calibrated
  probabilities, and the SVM only has a margin.
* If a model cannot be loaded it is reported as unavailable. Nothing is
  faked.

Only load pickle/joblib artifacts from a trusted source.
"""
from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

import config as cfg


# ---------------------------------------------------------------------------
# Errors and validation
# ---------------------------------------------------------------------------
class ValidationError(ValueError):
    """Raised when customer input is missing or invalid. `.problems` lists all issues."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class ModelUnavailableError(RuntimeError):
    """Raised when a model was requested but could not be loaded."""


class ArtifactError(RuntimeError):
    """Raised when a saved data artifact is missing or inconsistent with the others."""


def _as_number(value: Any, field_name: str, problems: list[str]) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        problems.append(f"{field_name} is required.")
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{field_name} must be a number.")
        return None
    if not np.isfinite(number):
        problems.append(f"{field_name} must be a finite number.")
        return None
    return number


def validate_customer(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate raw customer fields.

    Returns (clean_customer, warnings). Raises ValidationError listing every
    problem found. Warnings are for values outside the range the models saw in
    training (still allowed, but the prediction is an extrapolation).
    """
    problems: list[str] = []
    warn: list[str] = []
    clean: dict[str, Any] = {}

    for name in ("CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"):
        number = _as_number(raw.get(name), name, problems)
        if number is None:
            continue
        low, high = cfg.HARD_LIMITS[name]
        if not (low <= number <= high):
            problems.append(f"{name} must be between {low:,} and {high:,} (got {number:,.2f}).")
            continue
        if name in ("CreditScore", "Age", "Tenure", "NumOfProducts") and number != int(number):
            problems.append(f"{name} must be a whole number.")
            continue
        clean[name] = int(number) if name in ("CreditScore", "Age", "Tenure", "NumOfProducts") else float(number)

    for name in ("HasCrCard", "IsActiveMember"):
        number = _as_number(raw.get(name), name, problems)
        if number is None:
            continue
        if number not in (0.0, 1.0):
            problems.append(f"{name} must be Yes/No (0 or 1).")
            continue
        clean[name] = int(number)

    geography = raw.get("Geography")
    if geography not in cfg.GEOGRAPHIES:
        problems.append(f"Geography must be one of {', '.join(cfg.GEOGRAPHIES)}.")
    else:
        clean["Geography"] = geography

    gender = raw.get("Gender")
    if gender not in cfg.GENDERS:
        problems.append(f"Gender must be one of {', '.join(cfg.GENDERS)}.")
    else:
        clean["Gender"] = gender

    if problems:
        raise ValidationError(problems)

    # Soft checks against what the training data actually covered.
    if clean["Age"] > 92:
        warn.append("Age is above the oldest customer in the training data (92). Treat the result with extra caution.")
    if clean["CreditScore"] < 350 or clean["CreditScore"] > 850:
        warn.append("Credit score is outside the 350 to 850 range seen in training.")
    if clean["Balance"] > 250_898.09:
        warn.append("Balance is above the highest balance seen in training (about 250,898).")
    if clean["EstimatedSalary"] > 199_992.48:
        warn.append("Salary is above the highest salary seen in training (about 199,992).")
    return clean, warn


def encode_customers(frame: pd.DataFrame) -> pd.DataFrame:
    """Raw customer columns -> model-ready feature frame (unscaled, saved column order).

    Mirrors src/FULL_EDA.py: one-hot Geography and Gender, then reindex to the
    saved training columns. Unknown categories raise instead of silently
    becoming all-zero rows.
    """
    missing = [c for c in ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                           "HasCrCard", "IsActiveMember", "EstimatedSalary", "Geography", "Gender"]
               if c not in frame.columns]
    if missing:
        raise ValidationError([f"Missing column(s): {', '.join(missing)}"])
    bad_geo = set(frame["Geography"].astype(str)) - set(cfg.GEOGRAPHIES)
    bad_gen = set(frame["Gender"].astype(str)) - set(cfg.GENDERS)
    if bad_geo or bad_gen:
        raise ValidationError([f"Unknown category value(s): {sorted(bad_geo | bad_gen)}"])
    encoded = pd.get_dummies(
        frame[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
               "IsActiveMember", "EstimatedSalary", "Geography", "Gender"]].astype(
            {"Geography": str, "Gender": str}),
        columns=["Geography", "Gender"],
    ).astype(float)
    encoded = encoded.reindex(columns=cfg.FEATURE_COLUMNS, fill_value=0.0)
    if encoded.isna().any().any():
        raise ValidationError(["Encoding produced missing values."])
    return encoded


def encode_customer(customer: dict[str, Any]) -> pd.DataFrame:
    """Validated customer dict -> one-row encoded frame."""
    return encode_customers(pd.DataFrame([customer]))


# ---------------------------------------------------------------------------
# Prediction result container
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    model_key: str
    model_name: str
    churn: bool
    score: float
    threshold: float
    score_kind: str                  # 'probability-like' | 'margin'
    percentile: float | None = None  # share of held-out customers with a lower score (0-100)
    threshold_percentile: float | None = None
    risk_tier: str = "Low"           # Low | Moderate | High

    @property
    def label(self) -> str:
        return "Likely to Churn" if self.churn else "Likely to Stay"

    @property
    def has_probability_scale(self) -> bool:
        """True when the score lives on a 0 to 1 scale (still not calibrated)."""
        return self.score_kind == "probability-like"


# ---------------------------------------------------------------------------
# Individual model implementations. Each takes an ENCODED, UNSCALED frame.
# ---------------------------------------------------------------------------
def _flatten_tree(root: dict) -> dict[str, np.ndarray]:
    """Nested dict tree -> parallel arrays so batches can be scored without recursion."""
    feature, cut, left, right, prob = [], [], [], [], []

    def add(node: dict) -> int:
        index = len(feature)
        feature.append(-1); cut.append(0.0); left.append(-1); right.append(-1)
        prob.append(float(node["prob"]))
        if "feature" in node:
            feature[index] = int(node["feature"])
            cut[index] = float(node["cut"])
            left[index] = add(node["left"])
            right[index] = add(node["right"])
        return index

    add(root)
    return {"feature": np.array(feature), "cut": np.array(cut), "left": np.array(left),
            "right": np.array(right), "prob": np.array(prob)}


def _score_flat_tree(tree: dict[str, np.ndarray], values: np.ndarray) -> np.ndarray:
    node = np.zeros(len(values), dtype=int)
    rows = np.arange(len(values))
    while True:
        feat = tree["feature"][node]
        active = feat >= 0
        if not active.any():
            break
        go_left = values[rows[active], feat[active]] <= tree["cut"][node[active]]
        node[active] = np.where(go_left, tree["left"][node[active]], tree["right"][node[active]])
    return tree["prob"][node]


class _Base:
    spec: cfg.ModelSpec
    threshold: float
    notes: list[str]

    def score(self, encoded: pd.DataFrame) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_class(self, scores: np.ndarray) -> np.ndarray:
        return (scores >= self.threshold).astype(int)


class _TreeModel(_Base):
    def __init__(self, spec: cfg.ModelSpec, path: Path):
        self.spec, self.notes = spec, []
        with open(path, "rb") as handle:
            artifact = pickle.load(handle)
        self.feature_columns = list(artifact["feature_columns"])
        if self.feature_columns != cfg.FEATURE_COLUMNS:
            raise ValueError("Saved feature order differs from the app schema.")
        self.threshold = float(artifact["threshold"])
        self.trees = [_flatten_tree(root) for root in artifact["tree_roots"]]
        self.model_type = artifact.get("model_type", spec.kind)

    def score(self, encoded: pd.DataFrame) -> np.ndarray:
        values = encoded[self.feature_columns].to_numpy(dtype=float)
        return np.mean([_score_flat_tree(tree, values) for tree in self.trees], axis=0)


class _LogisticModel(_Base):
    def __init__(self, spec: cfg.ModelSpec, path: Path):
        self.spec, self.notes = spec, []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            artifact = joblib.load(path)
        for item in caught:
            if "InconsistentVersion" in item.category.__name__:
                self.notes.append(
                    "The saved scaler was pickled with a different scikit-learn version. "
                    "Only its mean and scale arrays are used, and they are cross-checked in the tests.")
        self.feature_columns = list(artifact["feature_columns"])
        if self.feature_columns != cfg.FEATURE_COLUMNS:
            raise ValueError("Saved feature order differs from the app schema.")
        self.scale_columns = list(artifact["scale_columns"])
        self.weights = np.asarray(artifact["weights"], dtype=float)
        self.bias = float(artifact["bias"])
        self.threshold = float(artifact["threshold"])
        self.mean = np.asarray(artifact["scaler"].mean_, dtype=float)
        self.scale = np.asarray(artifact["scaler"].scale_, dtype=float)
        if self.weights.shape != (len(self.feature_columns),):
            raise ValueError("Weight vector length does not match the feature list.")

    def _scaled(self, encoded: pd.DataFrame) -> np.ndarray:
        frame = encoded[self.feature_columns].copy()
        frame[self.scale_columns] = (frame[self.scale_columns].to_numpy(dtype=float) - self.mean) / self.scale
        return frame.to_numpy(dtype=float)

    def score(self, encoded: pd.DataFrame) -> np.ndarray:
        z = np.clip(self._scaled(encoded) @ self.weights + self.bias, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))


class _SVMModel(_Base):
    def __init__(self, spec: cfg.ModelSpec, path: Path):
        self.spec, self.notes = spec, []
        params = json.loads(Path(path).read_text(encoding="utf-8"))
        self.feature_columns = list(params["feature_names"])
        if self.feature_columns != cfg.FEATURE_COLUMNS:
            raise ValueError("Saved feature order differs from the app schema.")
        self.weights = np.asarray(params["weights"], dtype=float)
        self.bias = float(params["bias"])
        self.threshold = float(params["threshold"])
        self.mean = np.asarray(params["scaler_mean"], dtype=float)
        self.scale = np.asarray(params["scaler_scale"], dtype=float)
        self.notes.append("The SVM returns a decision margin. It is not a probability and is shown as a ranking score only.")

    def _scaled(self, encoded: pd.DataFrame) -> np.ndarray:
        return (encoded[self.feature_columns].to_numpy(dtype=float) - self.mean) / self.scale

    def score(self, encoded: pd.DataFrame) -> np.ndarray:
        return self._scaled(encoded) @ self.weights + self.bias


class _KNNModel(_Base):
    """KNN rebuilt from the saved training split and validation-selected settings."""

    def __init__(self, spec: cfg.ModelSpec, outputs: Path):
        self.spec, self.notes = spec, []
        settings = pd.read_csv(outputs / cfg.KNN_SETTINGS).iloc[0]
        if str(settings["Features"]) != "all" or str(settings["Scaling"]) != "standard":
            raise ValueError("Unsupported saved KNN settings (expected all features, standard scaling).")
        self.k = int(settings["K"])
        self.voting = str(settings["Voting"])
        self.threshold = float(settings["Threshold"])
        balance = str(settings["Balance"])

        train = pd.read_csv(outputs / cfg.X_TRAIN_ENCODED)
        labels = pd.read_csv(outputs / cfg.Y_TRAIN).iloc[:, 0].to_numpy(dtype=int)
        if list(train.columns) != cfg.FEATURE_COLUMNS:
            raise ValueError("Training columns differ from the app schema.")
        if len(train) != len(labels):
            raise ValueError("Training features and labels have different lengths.")
        self.feature_columns = list(train.columns)
        self.scale_columns = list(cfg.SCALE_COLUMNS)

        # Scaler statistics from ALL training customers (population std, like StandardScaler).
        numeric = train[self.scale_columns].to_numpy(dtype=float)
        self.mean = numeric.mean(axis=0)
        self.scale = numeric.std(axis=0)
        reference = train.copy()
        reference[self.scale_columns] = (numeric - self.mean) / self.scale
        reference = reference.to_numpy(dtype=float)

        if balance == "undersampled":
            rng = np.random.default_rng(42)
            churn = np.where(labels == 1)[0]
            stayed = np.where(labels == 0)[0]
            chosen = rng.choice(stayed, size=len(churn), replace=False)
            selected = np.concatenate([churn, chosen])
            rng.shuffle(selected)
            reference, labels = reference[selected], labels[selected]
        self.reference, self.labels = reference, labels
        self.notes.append(
            f"KNN is rebuilt at start-up from {len(labels):,} saved training customers "
            f"(k={self.k}, {self.voting} vote, undersampled to balance classes).")

    def _scaled(self, encoded: pd.DataFrame) -> np.ndarray:
        frame = encoded[self.feature_columns].copy()
        frame[self.scale_columns] = (frame[self.scale_columns].to_numpy(dtype=float) - self.mean) / self.scale
        return frame.to_numpy(dtype=float)

    def score(self, encoded: pd.DataFrame) -> np.ndarray:
        queries = self._scaled(encoded)
        out = np.empty(len(queries))
        for start in range(0, len(queries), 250):
            chunk = queries[start:start + 250]
            dist = np.sqrt(((chunk[:, None, :] - self.reference[None, :, :]) ** 2).sum(axis=2))
            order = np.argsort(dist, axis=1)[:, :self.k]
            for i, idx in enumerate(order):
                lab, d = self.labels[idx], dist[i, idx]
                if self.voting == "uniform":
                    out[start + i] = float(np.mean(lab))
                elif np.any(d == 0):
                    out[start + i] = float(np.mean(lab[d == 0]))
                else:
                    w = 1.0 / (d + 1e-8)
                    out[start + i] = float(np.sum(w * lab) / np.sum(w))
        return out


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
@dataclass
class ModelStatus:
    key: str
    name: str
    available: bool
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    threshold: float | None = None


class ModelService:
    """Loads every model once and exposes validated single and batch inference."""

    def __init__(self, outputs_dir: Path | str | None = None):
        self.outputs = Path(outputs_dir) if outputs_dir else cfg.OUTPUTS_DIR
        self.models: dict[str, _Base] = {}
        self.status: dict[str, ModelStatus] = {}
        self._test_scores: dict[str, np.ndarray] = {}
        for key in cfg.MODEL_ORDER:
            self._load(key)
        for key in self.models:
            self._load_reference_scores(key)

    # -- loading ----------------------------------------------------------
    def _load(self, key: str) -> None:
        spec = cfg.MODEL_SPECS[key]
        try:
            if spec.kind == "knn":
                model: _Base = _KNNModel(spec, self.outputs)
            else:
                path = self.outputs / spec.artifact
                if not path.exists():
                    raise FileNotFoundError(f"Missing model file: {spec.artifact}")
                model = {"tree": _TreeModel, "forest": _TreeModel,
                         "logistic": _LogisticModel, "svm": _SVMModel}[spec.kind](spec, path)
            self.models[key] = model
            self.status[key] = ModelStatus(key, spec.name, True, None, list(model.notes), model.threshold)
        except Exception as exc:  # reported in the UI, never swallowed silently
            self.status[key] = ModelStatus(key, spec.name, False, f"{type(exc).__name__}: {exc}")

    def _load_reference_scores(self, key: str) -> None:
        """Held-out scores, used only to rank a customer against the test set."""
        spec = cfg.MODEL_SPECS[key]
        path = self.outputs / spec.predictions
        if path.exists():
            try:
                self._test_scores[key] = pd.read_csv(path)[spec.pred_cols[2]].to_numpy(dtype=float)
            except Exception as exc:
                self.status[key].notes.append(f"Could not load held-out scores ({exc}). Score ranking is hidden.")

    # -- queries ----------------------------------------------------------
    def available_keys(self) -> list[str]:
        return [k for k in cfg.MODEL_ORDER if self.status[k].available]

    def is_available(self, key: str) -> bool:
        return key in self.models

    # -- inference --------------------------------------------------------
    def _result(self, key: str, score: float) -> PredictionResult:
        model = self.models[key]
        spec = cfg.MODEL_SPECS[key]
        churn = bool(score >= model.threshold)
        pct = thr_pct = None
        reference = self._test_scores.get(key)
        if reference is not None and len(reference):
            pct = float(np.mean(reference < score) * 100)
            thr_pct = float(np.mean(reference < model.threshold) * 100)
        if churn:
            tier = "High"
        elif pct is not None and thr_pct is not None and pct >= thr_pct - cfg.MODERATE_BAND_PCT_POINTS:
            tier = "Moderate"
        else:
            tier = "Low"
        return PredictionResult(key, spec.name, churn, float(score), float(model.threshold),
                                spec.score_kind, pct, thr_pct, tier)

    def predict_encoded(self, key: str, encoded: pd.DataFrame) -> list[PredictionResult]:
        if key not in self.models:
            reason = self.status.get(key).error if key in self.status else "unknown model"
            raise ModelUnavailableError(f"{cfg.MODEL_SPECS[key].name} is unavailable ({reason}).")
        scores = np.asarray(self.models[key].score(encoded), dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"{cfg.MODEL_SPECS[key].name} produced a non-finite score.")
        return [self._result(key, s) for s in scores]

    def predict(self, key: str, customer: dict[str, Any]) -> PredictionResult:
        clean, _ = validate_customer(customer)
        return self.predict_encoded(key, encode_customer(clean))[0]

    def predict_all_encoded(self, encoded: pd.DataFrame) -> dict[str, PredictionResult | str]:
        """Score ONE encoded (unscaled) row with every model.

        Failed/unavailable models map to an error string, never a fake result.
        """
        if len(encoded) != 1:
            raise ValueError("predict_all_encoded expects exactly one row.")
        out: dict[str, PredictionResult | str] = {}
        for key in cfg.MODEL_ORDER:
            try:
                out[key] = self.predict_encoded(key, encoded)[0]
            except Exception as exc:
                out[key] = f"{type(exc).__name__}: {exc}"
        return out

    def predict_all(self, customer: dict[str, Any]) -> dict[str, PredictionResult | str]:
        """Validate a raw customer, encode it, and run every model."""
        clean, _ = validate_customer(customer)
        return self.predict_all_encoded(encode_customer(clean))

    def raw_scores(self, key: str, encoded: pd.DataFrame) -> np.ndarray:
        """Scores without result wrapping (used for permutation importance)."""
        return np.asarray(self.models[key].score(encoded), dtype=float)


# ---------------------------------------------------------------------------
# Evaluation data (all read from saved project outputs, never invented)
# ---------------------------------------------------------------------------
def load_dataset(path: Path | None = None) -> pd.DataFrame | None:
    path = path or cfg.DATA_FILE
    return pd.read_csv(path) if Path(path).exists() else None


def load_predictions(key: str, outputs: Path | None = None) -> pd.DataFrame | None:
    """Held-out predictions standardised to y_true, y_pred, y_score."""
    outputs = outputs or cfg.OUTPUTS_DIR
    spec = cfg.MODEL_SPECS[key]
    path = outputs / spec.predictions
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if not set(spec.pred_cols).issubset(frame.columns):
        return None
    keep = (["CustomerId"] if "CustomerId" in frame.columns else []) + list(spec.pred_cols)
    frame = frame[keep].copy()
    frame.columns = (["CustomerId"] if "CustomerId" in keep else []) + ["y_true", "y_pred", "y_score"]
    return frame


def metrics_from_predictions(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(frame.y_true, frame.y_pred),
        "Precision": precision_score(frame.y_true, frame.y_pred, zero_division=0),
        "Recall": recall_score(frame.y_true, frame.y_pred, zero_division=0),
        "F1": f1_score(frame.y_true, frame.y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(frame.y_true, frame.y_score),
    }


def load_metrics(outputs: Path | None = None) -> pd.DataFrame | None:
    """Verified held-out metrics, one row per model (Model, Accuracy, ..., TN, FP, FN, TP)."""
    outputs = outputs or cfg.OUTPUTS_DIR
    path = outputs / cfg.COMPARISON_METRICS
    if path.exists():
        table = pd.read_csv(path)
        # The comparison export uses short names ("KNN", "SVM"). Use the app's names everywhere.
        aliases = {"KNN": "knn", "Logistic Regression": "logistic", "Decision Tree": "tree",
                   "Random Forest": "forest", "SVM": "svm"}
        table["Model"] = table["Model"].map(lambda m: cfg.MODEL_SPECS[aliases[m]].name if m in aliases else m)
        return table
    rows = []  # fall back to recomputing from the saved held-out predictions
    for key in cfg.MODEL_ORDER:
        pred = load_predictions(key, outputs)
        if pred is None:
            continue
        tn, fp, fn, tp = confusion_matrix(pred.y_true, pred.y_pred, labels=[0, 1]).ravel()
        rows.append({"Model": cfg.MODEL_SPECS[key].name, **metrics_from_predictions(pred),
                     "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    return pd.DataFrame(rows) if rows else None


def roc_points(key: str, outputs: Path | None = None):
    pred = load_predictions(key, outputs)
    if pred is None:
        return None
    fpr, tpr, _ = roc_curve(pred.y_true, pred.y_score)
    return fpr, tpr, roc_auc_score(pred.y_true, pred.y_score)


def permutation_importance(service: ModelService, key: str, repeats: int = 3, seed: int = 42,
                           outputs: Path | None = None) -> pd.DataFrame | None:
    """Drop in held-out ROC-AUC when one feature column is shuffled.

    Describes what the MODEL relies on. It does not say why customers leave.
    One-hot Geography and Gender columns are shuffled together as one feature.
    """
    outputs = outputs or service.outputs
    x_path, y_path = outputs / cfg.X_TEST_ENCODED, outputs / cfg.Y_TEST
    if key not in service.models or not (x_path.exists() and y_path.exists()):
        return None
    X = pd.read_csv(x_path)[cfg.FEATURE_COLUMNS]
    y = pd.read_csv(y_path).iloc[:, 0].to_numpy(dtype=int)
    base = roc_auc_score(y, service.raw_scores(key, X))
    groups = {c: [c] for c in cfg.FEATURE_COLUMNS if not c.startswith(("Geography_", "Gender_"))}
    groups["Geography"] = [c for c in cfg.FEATURE_COLUMNS if c.startswith("Geography_")]
    groups["Gender"] = [c for c in cfg.FEATURE_COLUMNS if c.startswith("Gender_")]
    rng = np.random.default_rng(seed)
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[cols] = X[cols].to_numpy()[order]
            drops.append(base - roc_auc_score(y, service.raw_scores(key, shuffled)))
        rows.append({"Feature": name, "Importance": float(np.mean(drops)), "Spread": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("Importance", ascending=False).reset_index(drop=True)


def linear_coefficients(service: ModelService, key: str) -> pd.DataFrame | None:
    """Standardised-feature weights for the two linear models (sign shows direction)."""
    if key not in ("logistic", "svm") or key not in service.models:
        return None
    model = service.models[key]
    return pd.DataFrame({"Feature": model.feature_columns, "Weight": model.weights})


def pr_points(key: str, outputs: Path | None = None):
    """Precision-recall curve points and average precision on the held-out set."""
    pred = load_predictions(key, outputs)
    if pred is None:
        return None
    precision, recall, _ = precision_recall_curve(pred.y_true, pred.y_score)
    return recall, precision, average_precision_score(pred.y_true, pred.y_score)


# ---------------------------------------------------------------------------
# Held-out test split with aligned labels (Test Dataset Explorer)
# ---------------------------------------------------------------------------
TARGET_LABELS = {0: "Stayed", 1: "Churned"}   # Exited: 0 = stayed, 1 = left (README, FULL_EDA.py)


@dataclass
class TestSplit:
    """The saved 20% held-out split. Row ``i`` of every member describes the same customer."""
    raw: pd.DataFrame          # original customer fields (readable strings) for display
    encoded: pd.DataFrame      # model-ready features, saved training order, unscaled
    y: np.ndarray              # ground-truth Exited label (0 stayed / 1 churned)
    customer_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def load_test_split(outputs: Path | None = None, data_file: Path | None = None) -> TestSplit:
    """Load the held-out split and PROVE that features, labels and identities line up.

    Sources: the raw dataset, ``comparison/canonical_test_manifest.csv`` (which
    stores each test customer's CustomerId, label and raw row position), the
    saved ``X_test_encoded.csv`` and ``y_test.csv``. Nothing is fitted or
    modified here. Raises ArtifactError with a plain-language reason if any
    file is missing or any alignment check fails.
    """
    outputs = Path(outputs) if outputs else cfg.OUTPUTS_DIR
    data_file = Path(data_file) if data_file else cfg.DATA_FILE
    manifest_path = outputs / "comparison" / "canonical_test_manifest.csv"
    needed = {"dataset": data_file, "test manifest": manifest_path,
              "encoded test features": outputs / cfg.X_TEST_ENCODED, "test labels": outputs / cfg.Y_TEST}
    missing = [f"{name} ({path.name})" for name, path in needed.items() if not path.exists()]
    if missing:
        raise ArtifactError("Missing file(s) needed for the held-out test split: " + ", ".join(missing) + ".")

    dataset = pd.read_csv(data_file)
    manifest = pd.read_csv(manifest_path)
    x_test = pd.read_csv(outputs / cfg.X_TEST_ENCODED)
    y_test = pd.read_csv(outputs / cfg.Y_TEST).iloc[:, 0].to_numpy(dtype=int)

    if not (len(manifest) == len(x_test) == len(y_test)):
        raise ArtifactError(f"Test artifacts disagree on size (manifest {len(manifest)}, features {len(x_test)}, labels {len(y_test)}).")
    if list(x_test.columns) != cfg.FEATURE_COLUMNS:
        raise ArtifactError("Saved test features do not match the app's feature schema.")
    raw = dataset.iloc[manifest["source_row_position"].to_numpy()].reset_index(drop=True)
    if not (raw["CustomerId"].to_numpy() == manifest["CustomerId"].to_numpy()).all():
        raise ArtifactError("Test manifest CustomerIds do not match the dataset rows they point to.")
    if not (raw["Exited"].to_numpy(dtype=int) == y_test).all() or not (manifest["y_true"].to_numpy(dtype=int) == y_test).all():
        raise ArtifactError("Test labels are not aligned with the test records.")
    if set(np.unique(y_test)) - set(TARGET_LABELS):
        raise ArtifactError(f"Unexpected target label values: {sorted(set(np.unique(y_test)) - set(TARGET_LABELS))}.")
    encoded_from_raw = encode_customers(raw)
    if not np.allclose(encoded_from_raw.to_numpy(), x_test.to_numpy()):
        raise ArtifactError("Saved encoded test features differ from re-encoding the raw test rows.")
    return TestSplit(raw=raw, encoded=x_test, y=y_test, customer_ids=raw["CustomerId"].to_numpy())


def saved_prediction(key: str, customer_id: int, outputs: Path | None = None) -> dict[str, float] | None:
    """The prediction the training/evaluation scripts exported for this test customer (or None)."""
    pred = load_predictions(key, outputs)
    if pred is None or "CustomerId" not in pred.columns:
        return None
    row = pred[pred.CustomerId == customer_id]
    if row.empty:
        return None
    row = row.iloc[0]
    return {"y_pred": int(row.y_pred), "y_score": float(row.y_score)}
