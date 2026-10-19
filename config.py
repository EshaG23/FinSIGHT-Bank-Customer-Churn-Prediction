"""Central configuration for the FinSIGHT Streamlit app.

All paths are relative to the project root and can be overridden with
environment variables, so nothing here is machine specific.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("FINSIGHT_ROOT", Path(__file__).resolve().parent)).resolve()
OUTPUTS_DIR = Path(os.environ.get("FINSIGHT_OUTPUTS", PROJECT_ROOT / "outputs")).resolve()
DATA_FILE = Path(os.environ.get("FINSIGHT_DATA", PROJECT_ROOT / "data" / "Churn_Modelling.csv")).resolve()
ASSETS_DIR = PROJECT_ROOT / "assets"

# ---------------------------------------------------------------------------
# Feature schema. This is the exact schema produced by src/FULL_EDA.py:
#   drop RowNumber, CustomerId, Surname, Exited
#   one-hot encode Geography and Gender (get_dummies, alphabetical order)
#   column order below == outputs/feature_columns.pkl
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary",
    "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male",
]
SCALE_COLUMNS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
GEOGRAPHIES = ["France", "Germany", "Spain"]
GENDERS = ["Female", "Male"]

# Hard limits used for validation (plausible banking domain).
# Soft ranges (observed in the training data) trigger a warning, not an error.
HARD_LIMITS = {
    "CreditScore": (300, 900),
    "Age": (18, 100),
    "Tenure": (0, 10),
    "Balance": (0.0, 1_000_000.0),
    "NumOfProducts": (1, 4),
    "EstimatedSalary": (0.0, 1_000_000.0),
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    kind: str                 # 'tree' | 'forest' | 'logistic' | 'svm' | 'knn'
    artifact: str | None      # path relative to OUTPUTS_DIR (None = rebuilt from training data)
    predictions: str          # held-out prediction export, relative to OUTPUTS_DIR
    pred_cols: tuple[str, str, str]   # (actual, predicted, score) column names in that export
    score_kind: str           # 'probability-like' or 'margin'
    blurb: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "knn": ModelSpec(
        "knn", "K-Nearest Neighbors", "knn", None,
        "knn_improved/test_predictions.csv", ("Actual", "Predicted", "Churn_Score"),
        "probability-like",
        "Looks at the most similar customers in the training data and takes a vote.",
    ),
    "logistic": ModelSpec(
        "logistic", "Logistic Regression", "logistic", "logistic_results/logistic_model.pkl",
        "logistic_results/logistic_predictions.csv", ("Actual", "Predicted", "Churn_Probability"),
        "probability-like",
        "A weighted sum of customer features squashed into a 0 to 1 score.",
    ),
    "tree": ModelSpec(
        "tree", "Decision Tree", "tree", "decision_tree_scratch_entropy/model.pkl",
        "decision_tree_scratch_entropy/predictions.csv", ("y_true", "y_pred", "y_score"),
        "probability-like",
        "One tree of yes/no questions about the customer. Easy to explain.",
    ),
    "forest": ModelSpec(
        "forest", "Random Forest", "forest", "random_forest_scratch_entropy/model.pkl",
        "random_forest_scratch_entropy/predictions.csv", ("y_true", "y_pred", "y_score"),
        "probability-like",
        "80 decision trees that each vote, averaged for a steadier score.",
    ),
    "svm": ModelSpec(
        "svm", "Support Vector Machine", "svm", "svm_scratch/model_parameters.json",
        "svm_scratch/predictions.csv", ("Actual", "Predicted", "DecisionScore"),
        "margin",
        "A linear boundary between leavers and stayers. Outputs a margin, not a probability.",
    ),
}
MODEL_ORDER = ["knn", "logistic", "tree", "forest", "svm"]

# Shared artifacts
COMPARISON_METRICS = "comparison/comparison_metrics.csv"
KNN_SETTINGS = "knn_improved/selected_settings.csv"
X_TRAIN_ENCODED = "X_train_encoded.csv"
Y_TRAIN = "y_train.csv"
X_TEST_ENCODED = "X_test_encoded.csv"
Y_TEST = "y_test.csv"
GLOBAL_FEATURE_COLUMNS = "feature_columns.pkl"
GLOBAL_SCALER = "scaler.pkl"

# Risk tier: a customer below the model's alert line is "Moderate" when their
# score ranks within this many percentile points of that line.
MODERATE_BAND_PCT_POINTS = 15.0

# Colours live in theme.py (light and dark palettes).
