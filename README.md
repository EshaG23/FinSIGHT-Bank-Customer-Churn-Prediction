
# FinSIGHT — Intelligent Banking Customer Churn Prediction and Retention Analytics

FinSIGHT is a Machine Learning Lab mini-project that predicts whether a bank customer may leave the bank (**customer churn**) using the public Kaggle Churn Modelling dataset.

The project compares five machine learning algorithms, provides exploratory data analysis and model-performance dashboards, and offers an interactive Streamlit application for examining known test customers and predicting churn for new customer profiles.

> **Run the interactive application:** See [README_APP.md](README_APP.md) for additional application details. This README covers the complete project, including training scripts, evaluation, comparison, and the web application.

---

## 1. Dataset and Objective

- **Dataset:** `data/Churn_Modelling.csv` — 10,000 customer records and 14 original columns.
- **Target:** `Exited` (`0` = stayed, `1` = left).
- **Excluded prediction features:** `RowNumber`, `CustomerId`, and `Surname`.
- **Input features:** `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
- **Class distribution:** 7,963 stayed (79.63%); 2,037 left (20.37%).
- **Data quality:** No missing values or duplicate rows in the dataset inspected.

`CustomerId` may be retained in comparison metadata to align predictions for the same test customers. It is not used to train the models.

---

## 2. Machine Learning Models

All five algorithms are implemented in Python with **from-scratch training and prediction logic**, rather than calling the equivalent ready-made scikit-learn classifiers.

| Model | Training Script | Approach |
|---|---|---|
| K-Nearest Neighbors (KNN) | `src/KNN.py` | Euclidean distance, nearest-neighbor voting, validation tuning |
| Logistic Regression | `src/Logistic_Regression.py` | Sigmoid, gradient descent, regularization, validation tuning |
| Decision Tree | `src/Decision_Tree.py` | Entropy and information gain, recursive binary threshold splits |
| Random Forest | `src/Random_Forest.py` | Bootstrap sampling, multiple entropy-based trees, ensemble predictions |
| Support Vector Machine (SVM) | `src/SVM.py` | Linear SVM with hinge-loss optimization and validation tuning |

The entropy-based Decision Tree uses binary threshold splits for numerical banking features. It is inspired by ID3 concepts but is not an exact implementation of categorical ID3.

---

## 3. Project Structure

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
├── regenerate_predictions.py    # Prediction and identity verification
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
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── requirements-comparison.txt
├── README.md
├── README_APP.md
└── README_COMPARISON.md
```

Some auxiliary folders and files have been omitted for readability.

Saved artifacts under `outputs/` are required for inference and comparison. Do not remove them from the deployed project.

---

## 4. Installation and Local Execution

Open the project root folder (the folder containing `app.py`) in VS Code or a terminal.

### Step 1: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 2: Start the Streamlit Application

```powershell
python -m streamlit run app.py
```

The application normally opens at:

`http://localhost:8501`

### Step 3: Run Individual ML Models (Optional)

To rerun data preparation and model training:

```powershell
python src/FULL_EDA.py
python src/KNN.py
python src/Logistic_Regression.py
python src/Decision_Tree.py
python src/Random_Forest.py
python src/SVM.py
```

The scripts may take time, especially the from-scratch KNN and Random Forest implementations.

Running the scripts can regenerate results and model artifacts. Maintain a backup if preserving an exact release is important.

---

## 5. Exploratory Data Analysis and Preprocessing

The `src/FULL_EDA.py` script performs the following operations:

1. Loads the original banking dataset.
2. Inspects data using pandas summaries and visualizations.
3. Checks missing values, duplicates, distributions, correlations, and churn-group averages.
4. Excludes `RowNumber`, `CustomerId`, and `Surname` from prediction features without changing the original CSV.
5. One-hot encodes `Geography` and `Gender`, producing 13 encoded input columns.
6. Performs a stratified 80:20 train-test split using `random_state=42`.
7. Creates 8,000 training records and 2,000 held-out testing records.
8. Fits `StandardScaler` on training data only.
9. Transforms test data using the same fitted scaler.
10. Saves encoded and scaled datasets, labels, scaler, and feature metadata under `outputs/`.

Some models fit their own scalers during training and validation.

To prevent data leakage, preprocessing must not learn from the held-out testing data.

---

## 6. Five-Model Performance Evaluation

The following results were recorded on the held-out test dataset using the finalized model configurations.

Precision, recall, and F1-score refer to the positive churn class (`Exited = 1`).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| KNN | 74.20% | 42.65% | **77.64%** | 55.05% | 0.8389 |
| Logistic Regression | 77.85% | 46.33% | 55.77% | 50.61% | 0.7773 |
| Decision Tree | 82.00% | 54.65% | 67.81% | **60.53%** | 0.8425 |
| Random Forest | **82.90%** | **57.08%** | 64.37% | 60.51% | **0.8617** |
| SVM | 75.40% | 42.66% | 60.69% | 50.10% | 0.7670 |

### Performance Interpretation

- **KNN** achieves the highest churn recall, identifying approximately 78% of actual churners.
- **Random Forest** achieves the highest accuracy, precision, and ROC-AUC.
- **Decision Tree** achieves the highest F1-score, marginally above Random Forest.
- **Logistic Regression** provides a linear classification baseline.
- **SVM** demonstrates binary classification using a margin-based learning approach.

The models use different validation-selected prediction thresholds.

These results are not guarantees of future performance. Since the test set was consulted during iterative development, a fresh independent evaluation would be required for a fully unbiased final performance estimate.

For detailed comparisons and visualizations, see [README_COMPARISON.md](README_COMPARISON.md) and `outputs/comparison/`.

---

## 7. Streamlit Web Application

FinSIGHT provides an interactive Streamlit application with multiple features.

### Home Dashboard

- Project overview and introduction.
- Summary of the five machine learning models.
- Banking customer churn analytics.

### Known Test Customer Explorer

- Select a customer from the 2,000 held-out test records.
- View the actual `Exited` value.
- Compare predictions from all five machine learning models.
- Identify correct and incorrect predictions.

### New Customer Prediction

- Enter customer information manually.
- Generate predictions using all five trained models.
- Display whether each model predicts customer churn or retention.
- Present model scores with appropriate interpretation.

For a completely new customer, the actual churn outcome is unknown.

### Model Comparison Dashboard

- Accuracy comparison.
- Precision comparison.
- Recall comparison.
- F1-score comparison.
- ROC-AUC comparison.
- Confusion matrices.
- ROC curves.

### Insights and Retention Analytics

- Exploratory customer behaviour analysis.
- Churn distribution and feature analysis.
- Illustrative customer-retention considerations.

### Additional Interface Features

- Prediction history.
- Light and dark themes.
- Customer input validation.
- Interactive navigation and visualizations.

The application loads saved trained model artifacts and applies each model's required feature order, encoding, scaling, and prediction threshold.

It does not retrain all five models every time a prediction is requested.

**Important:** The raw decision margin of a linear SVM is not automatically a churn probability. Other model scores should also not be described as calibrated probabilities without appropriate calibration checks.

---

## 8. Model Comparison and Verification

The project includes a comparison workflow that verifies predictions across all five models.

It performs:

- Customer identity alignment using `CustomerId` metadata.
- Verification of actual target labels.
- Evaluation of the same 2,000 test customers.
- Performance metric comparisons.
- Confusion matrix visualization.
- ROC curve visualization.
- Generation of a comparison dashboard.

### Run the Comparison

Install comparison dependencies:

```powershell
python -m pip install -r requirements-comparison.txt
```

Generate and verify predictions:

```powershell
python regenerate_predictions.py
python compare_models.py
```

Run these commands from the project root with the required saved model artifacts available.

The HTML comparison dashboard is generated at:

`outputs/comparison/dashboard.html`

### Run Automated Tests

```powershell
python -m pytest -q
```

Install the test dependencies if necessary.

---

## 9. Deployment

The FinSIGHT application can be deployed using Streamlit Community Cloud.

### Deployment Steps

1. Push the complete working project to GitHub.
2. Include the required saved model artifacts under `outputs/`.
3. Open [Streamlit Community Cloud](https://share.streamlit.io).
4. Connect the GitHub repository.
5. Select the `main` branch.
6. Set the main application file to `app.py`.
7. Choose a compatible Python environment and install dependencies using `requirements.txt`.
8. Deploy the application.
9. Verify the homepage, customer prediction modes, model comparison, and insights on the deployed website.

### Security Considerations

Do not commit:

- `.env` files or passwords.
- `.streamlit/secrets.toml`.
- API keys or access tokens.
- Private customer information.
- Virtual environments.
- Cache folders and temporary files.

Required saved model artifacts must remain available for deployment.

---

## 10. Project Limitations

- FinSIGHT is an educational machine learning proof of concept based on a public Kaggle dataset.
- It is not a validated production banking system.
- The models identify associations useful for customer churn-risk screening.
- Predictions do not establish the causes of churn.
- Retention suggestions are illustrative and do not guarantee customer retention.
- New-customer predictions cannot immediately be declared correct or incorrect without knowing the actual outcome.
- Accuracy alone is insufficient when evaluating imbalanced datasets; precision, recall, and F1-score should also be considered.
- Saved pickle and joblib model artifacts should only be loaded from trusted sources and compatible environments.

---

## 11. Additional Documentation

| Resource | Description |
|---|---|
| [README_APP.md](README_APP.md) | Streamlit application setup, features, and interface |
| [README_COMPARISON.md](README_COMPARISON.md) | Five-model comparison, identity verification, and evaluation |
| `docs/screenshots/` | Application screenshots and visual evidence |
| `outputs/comparison/` | Comparison results, confusion matrices, ROC curves, and dashboard |
| `src/` | Machine learning training scripts and EDA implementation |

---

**FinSIGHT — Intelligent Banking Customer Churn Prediction and Retention Analytics**
