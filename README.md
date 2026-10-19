FinSIGHT — Intelligent Banking Customer Churn Prediction and Retention Analytics
FinSIGHT is a Machine Learning Lab mini-project that predicts whether a bank customer may leave the bank (customer churn) using the public Kaggle Churn Modelling dataset. It compares five machine learning algorithms, provides exploratory data analysis and model-performance dashboards, and offers an interactive Streamlit application for examining known test customers and predicting churn for new customer profiles.
> **Run the interactive application:** See [README_APP.md](README_APP.md) for additional application details. This README covers the complete project, including the training scripts, evaluation, comparison, and web app.
1. Dataset and objective
Dataset: `data/Churn_Modelling.csv` — 10,000 records and 14 original columns.
Target: `Exited` (`0` = stayed, `1` = left).
Excluded prediction features: `RowNumber`, `CustomerId`, and `Surname`.
Input features: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
Class distribution: 7,963 stayed (79.63%); 2,037 left (20.37%).
Data quality: No missing values or duplicate rows in the dataset inspected.
`CustomerId` may be retained in comparison metadata to align predictions for the same test customers. It is not used to train the models.
2. Machine learning models
All five algorithms are implemented in Python with from-scratch prediction/training logic rather than calling the equivalent ready-made scikit-learn classifier:
Model	Training script	Approach
K-Nearest Neighbors (KNN)	`src/KNN.py`	Euclidean distance, nearest-neighbor voting, validation tuning
Logistic Regression	`src/Logistic_Regression.py`	Sigmoid, gradient descent, regularization, validation tuning
Decision Tree	`src/Decision_Tree.py`	Entropy and information gain, recursive binary threshold splits
Random Forest	`src/Random_Forest.py`	Bootstrap samples, multiple entropy-based trees, ensemble predictions
Support Vector Machine (SVM)	`src/SVM.py`	Linear SVM with hinge-loss optimization and validation tuning
The entropy-based tree uses binary threshold splits for numeric banking features; it is inspired by ID3 concepts but is not an exact implementation of categorical ID3.
3. Project structure
```text
FinSIGHT/
├── app.py                       # Streamlit entry point
├── model_service.py             # Model loading and inference
├── config.py                    # Application configuration
├── analytics.py                 # Analytics helpers
├── state.py                     # Application state
├── theme.py                     # UI theme
├── ui.py                        # UI helpers
├── compare_models.py            # Five-model evaluation comparison
├── regenerate_predictions.py    # Prediction/identity verification
├── data/
│   └── Churn_Modelling.csv
├── src/
│   ├── FULL_EDA.py
│   ├── KNN.py
│   ├── Logistic_Regression.py
│   ├── Decision_Tree.py
│   ├── Random_Forest.py
│   └── SVM.py
├── outputs/                     # Processed data, model artifacts and results
│   ├── knn_improved/
│   ├── logistic_results/
│   ├── decision_tree_scratch_entropy/
│   ├── random_forest_scratch_entropy/
│   ├── svm_scratch/
│   ├── comparison/
│   └── identity_verification/
├── views/                       # Streamlit pages
├── assets/                      # Styling and assets
├── docs/screenshots/            # Application screenshots
├── tests/                       # Test suite
├── .streamlit/config.toml
├── requirements.txt
├── requirements-comparison.txt
├── README.md
├── README_APP.md
└── README_COMPARISON.md
```
Some auxiliary folders/files have been omitted for readability. Saved artifacts under `outputs/` are needed for inference and comparison; do not remove them from the deployed project.
4. Installation and local execution
Open the project root folder (the one containing `app.py`) in VS Code or a terminal.
Install dependencies:
```powershell
python -m pip install -r requirements.txt
```
Start the Streamlit application:
```powershell
python -m streamlit run app.py
```
Streamlit normally opens at `http://localhost:8501`.
To rerun the data preparation and individual training scripts (optional when the supplied artifacts already exist):
```powershell
python src/FULL_EDA.py
python src/KNN.py
python src/Logistic_Regression.py
python src/Decision_Tree.py
python src/Random_Forest.py
python src/SVM.py
```
The scripts may take time, especially the from-scratch KNN and Random Forest. Running model scripts can regenerate the corresponding results and artifacts; make a backup if preserving an exact release is important.
5. EDA and preprocessing
`src/FULL_EDA.py`:
Inspects data using pandas summaries and visualizations; checks missing values, duplicates, distributions, correlations, and churn-group averages.
Excludes `RowNumber`, `CustomerId`, and `Surname` from model features without changing the original CSV.
One-hot encodes `Geography` and `Gender` into 13 encoded input columns.
Produces a stratified 80:20 split (`random_state=42`): 8,000 training records and 2,000 held-out test records.
Fits `StandardScaler` on training data only, then transforms test data using that scaler.
Saves encoded/scaled features, labels, scaler and feature metadata under `outputs/`.
Some models fit their own scaler during training/validation. For valid evaluation, preprocessing must not learn from held-out test data.
6. Five-model evaluation
These are the recorded held-out test results for the finalized model configurations. Precision, recall and F1-score refer to the churn class (`Exited = 1`).
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
KNN	74.20%	42.65%	77.64%	55.05%	0.8389
Logistic Regression	77.85%	46.33%	55.77%	50.61%	0.7773
Decision Tree	82.00%	54.65%	67.81%	60.53%	0.8425
Random Forest	82.90%	57.08%	64.37%	60.51%	0.8617
SVM	75.40%	42.66%	60.69%	50.10%	0.7670
Interpretation: KNN has the highest churn recall; Random Forest has the highest accuracy, precision and ROC-AUC; Decision Tree has a marginally higher F1-score than Random Forest. These results reflect different validation-selected thresholds and should not be read as a guarantee of future outcomes. The test set was consulted during iterative development, so an independent evaluation would be needed for a fully unbiased final performance estimate.
For full details and comparison graphics, see README_COMPARISON.md and `outputs/comparison/`.
7. Streamlit application features
The application supports:
Home: Project overview and dashboard.
Known Test Customer Explorer: Inspect one of the 2,000 held-out records, view its known `Exited` value, and compare predictions across all five models.
New Customer Prediction: Enter a new customer's details and view separate predictions from all five models. The actual churn outcome for an unseen customer is unknown.
Model Comparison: Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices and ROC curves.
Insights and Retention: Exploratory findings and illustrative customer-retention considerations.
Prediction History and Theme: Session-level interaction features and light/dark interface options, subject to the application's implemented persistence behavior.
The app loads trained artifacts and applies each model's required feature order, encoding, scaling and classification threshold; it does not retrain five models on every prediction. Scores from a linear SVM decision margin are not automatically probabilities. Other model scores also should not be described as calibrated probabilities without calibration checks.
8. Comparison and verification
The project provides a comparison workflow that aligns model predictions with the same test customers using customer identity metadata, verifies the actual labels, and produces tables and charts.
```powershell
python -m pip install -r requirements-comparison.txt
python regenerate_predictions.py
python compare_models.py
```
Run these from the project root with the saved model artifacts in place. The HTML comparison dashboard is available at `outputs/comparison/dashboard.html`.
For the app's automated tests, install the test dependencies if necessary and run:
```powershell
python -m pytest -q
```
9. Deploying the application
The Streamlit app can be deployed on Streamlit Community Cloud from a GitHub repository:
Push the full working project, including required model artifacts, to GitHub.
Open Streamlit Community Cloud and connect the repository.
Select branch `main` and main file `app.py` (assuming it is at the repository root).
Use a Python/dependency environment compatible with the saved model artifacts.
Deploy, then verify the home page, known-customer mode, new-customer predictions, comparisons and insights on the public URL.
Do not commit `.env`, `.streamlit/secrets.toml`, passwords, access tokens, private data, virtual environments or cache folders. Retain required `outputs/` model artifacts for deployment.
10. Notes and limitations
This is an educational proof of concept based on a public dataset, not a validated production banking system.
The models identify associations useful for churn-risk screening; they do not establish the cause of churn or guarantee retention success.
New-customer predictions cannot be declared individually correct without a known outcome.
Accuracy alone is not sufficient on an imbalanced dataset; evaluate precision and recall together.
Saved pickle/joblib artifacts should only be loaded from trusted sources and compatible environments.
11. Further documentation
README_APP.md — Streamlit application setup and interface.
README_COMPARISON.md — Model comparison, identity checks and metrics.
`docs/screenshots/` — UI evidence and screenshots.
FinSIGHT — Intelligent Banking Customer Churn Prediction and Retention Analytics
