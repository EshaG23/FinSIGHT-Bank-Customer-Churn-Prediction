# FinSIGHT — Verified five-model comparison

## Start here
Open outputs/comparison/dashboard.html. No server or installation needed to
view the self-contained dashboard. It is now VERIFIED_BY_CUSTOMER_ID.

All five algorithms were used to generate fresh predictions while preserving
CustomerId from each raw input row BEFORE splitting, encoding and inference.
The original prediction CSVs were not simply assigned IDs after the fact.

Exactly the same 2,000 unique customer IDs and matching actual labels are
present across all five regenerated exports. All original prediction labels
and reported metrics remain unchanged, apart from negligible score rounding.

## Run in VS Code
Open this folder in a terminal:

    python -m pip install -r requirements-comparison.txt
    python regenerate_predictions.py
    python compare_models.py

Do NOT use --allow-legacy-order for these verified files.
regenerate_predictions.py performs prediction export; compare_models.py
validates those IDs and generates the complete dashboard and report files.
The source scripts under src/ are the supplied training implementations;
use the root regeneration script for the canonical ID-preserving exports.

## How regeneration works
- Rebuild raw train/test inputs using stratified 80:20 split and seed 42.
- Retain CustomerId as the dataframe index throughout the test-input path.
- Verify reconstructed train/test encoded feature values and labels against
  the shared CSV files before generating predictions.
- Decision Tree/Random Forest: run saved tree roots on rebuilt test inputs.
- Logistic Regression: use saved weights, bias and training scaler statistics.
- SVM: use saved weights, bias and training scaler statistics from JSON.
- KNN: reconstruct the classifier from its already selected validation settings
  and the training data, including the original undersampling procedure.
- Attach the RETAINED input IDs to these freshly computed prediction outputs.
- Check all previous predictions and scores reproduce before publishing exports.
- Run strict comparison, validating unique/exact IDs and labels for each model.

The existing thresholds and hyperparameters are preserved; there is no
additional tuning on the test set. Stored scaler numeric statistics are used
for LR inference; the uploaded scaler object may issue a sklearn version
warning when loaded, but regenerated scores were explicitly checked against
all previous scores within numerical tolerance.

## Results
Random Forest leads accuracy (82.90%), precision (57.08%) and ROC-AUC
(0.8617). KNN leads recall (77.64%). Decision Tree leads F1 (0.6053).
These describe this held-out set, not statistical significance or production
validation. Always-stayed baseline accuracy is 79.65%, with zero churn recall.
SVM scores are decision margins, not probabilities. Each model uses its own
validation-selected threshold. These scratch-model results belong to the
latest uploaded project and differ from the earlier sklearn Member 2 package.

## Files
- compare_models.py: single five-model comparison generator.
- regenerate_predictions.py: genuine prediction regeneration retaining IDs.
- outputs/comparison/dashboard.html: offline dashboard with embedded graphs.
- outputs/comparison/comparison_metrics.csv, best_by_metric.csv.
- outputs/comparison/: graphs, confusion matrices, ROC curves, strict audit.
- outputs/identity_verification/: regeneration audit and provenance.
- outputs/*/prediction CSVs: fresh predictions with actual retained CustomerIds.

The package includes original trusted model artifacts needed for regeneration.
Only load pickle/joblib files from trusted sources. Identity verification does
not itself certify training methodology or exclude every form of leakage.
