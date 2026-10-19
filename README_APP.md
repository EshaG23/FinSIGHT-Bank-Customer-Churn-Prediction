# FinSIGHT — Customer Churn Intelligence (Streamlit app)

## 1. Overview
FinSIGHT scores bank customers for churn risk with five from-scratch machine-learning models trained on the Kaggle
*Churn Modelling* dataset (10,000 customers; `Exited` 0 = stayed, 1 = churned). The app has two demonstration modes
(known held-out customers, and new manually entered customers), full light/dark themes, and evaluation pages that use only
the project's saved results.

## 2. Features
- **Overview Dashboard**: what churn is, KPIs (held-out records, test churn rate, models available, best model), churn distribution, model comparison, actual vs flagged churners, feature importance.
- **Test Dataset Explorer** (mode 1): *Generate Random Test Record* draws a customer from the 2,000-record held-out split, shows its real features and true label, runs all five models and marks each Correct/Incorrect.
- **New Customer Prediction** (mode 2): validated form for the 10 raw model inputs; all five models predict. Outcome is unknown, so nothing is marked correct/incorrect.
- **Model Comparison**: single-customer comparison (test record or new customer, clearly labelled) and overall test-set metrics + confusion matrices.
- **Performance & Insights**: metrics, churn distribution, confusion matrices, ROC, precision-recall, permutation importance (+ weights for linear models).
- **Retention Insights** (from the earlier app) and **Prediction History** (session only, clearable, exportable; no new-customer details stored).
- **Light / Dark mode**: switch in the sidebar and a button top-right of every page. Charts, tables, forms and dropdowns all follow it.

## 3. Technology
Python 3.10+, Streamlit ≥ 1.50 (developed on 1.66), Plotly, pandas, NumPy, scikit-learn (metrics only), joblib. Tests: pytest. Optional visual audit: Playwright.

## 4. Existing artifacts (unchanged)
| Model | Artifact | Notes |
|---|---|---|
| K-Nearest Neighbors | `outputs/knn_improved/selected_settings.csv` + saved train split | No pickle: rebuilt at start-up from saved training data with the validation-selected settings |
| Logistic Regression | `outputs/logistic_results/logistic_model.pkl` | weights, bias, scaler stats, threshold |
| Decision Tree | `outputs/decision_tree_scratch_entropy/model.pkl` | |
| Random Forest | `outputs/random_forest_scratch_entropy/model.pkl` | 80 trees |
| SVM | `outputs/svm_scratch/model_parameters.json` | outputs a margin, **not** a probability |

Shared: `feature_columns.pkl`, `scale_columns.pkl`, `scaler.pkl`, `X_/y_ train/test` CSVs, `comparison/comparison_metrics.csv` (verified metrics),
`comparison/canonical_test_manifest.csv` (CustomerId, label and raw row position of each test record).
Preprocessing is exactly the training one: one-hot Geography/Gender in the saved column order, scaling with the **saved training statistics** (never refit).
Each model keeps its own validation-chosen alert line, so predictions are not made at 0.5. Scores are raw, uncalibrated model outputs.

## 5. Install and run
```bash
python -m pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```
Run the tests: `python -m pytest -q`.

## 6. Demonstrate the test-dataset workflow
Sidebar → **Test Dataset Explorer** → **Generate Random Test Record**. Show the actual outcome, the five predictions, the correct/incorrect column and the
"X of 5" count; press again for another record. The *Inspect a specific record* expander loads a chosen row. A small integrity line confirms live predictions equal the saved evaluation export.
One record is an illustration, not an accuracy score; overall metrics are on Model Comparison.

## 7. Demonstrate new-customer prediction
Sidebar → **New Customer Prediction** → edit the fields → **Predict Customer Churn**. Compare per-model cards, probabilities (SVM shows a margin only), disagreement text,
rank chart, and model sensitivity. Then *Compare all five predictions* or *View model performance*.

## 8. Light / Dark mode
Use the **Dark mode** switch in the sidebar or the button at the top-right. The choice is kept in session state across reruns and page changes
(it resets to Light if the browser tab is reloaded, which starts a new session). Palettes live in `theme.py`; styles in `assets/styles.css`.

## 9. Verification performed
- `python -m pytest -q`: 102 tests (schema, preprocessing, split alignment, live-vs-saved score reproduction, failure handling, both demo flows, theme state/CSS/charts, WCAG contrast of design tokens).
- `python tools/visual_check.py` (needs `pip install playwright && playwright install chromium`): drives every page in both themes in headless Chromium, writes screenshots to `docs/screenshots/`, and measures text contrast, including chart labels and an open dropdown. Last run: no failures. It skips text drawn on coloured chart marks; those pairs are covered numerically by the unit tests.
- Not done: testing on real phones/tablets, other browsers (only Chromium), or screen readers. Responsive CSS exists but was only checked at desktop width.

## 10. Known limitations
- Streamlit's own theme is static, so the switch is implemented with CSS overrides on Streamlit 1.66's `data-testid` hooks; a major Streamlit upgrade may need selector tweaks (re-run `tools/visual_check.py`).
- Reloading the browser resets the theme to Light.
- KNN is rebuilt at start-up (a few seconds). Churn "probabilities" are uncalibrated; SVM has none.
- Pickle files must come from a trusted source.
- Existing `README.md` documents the original training scripts (Member 1 work).
