"""FinSIGHT - SVM from scratch (Member 3, integrated with FULL_EDA.py).
Run from the FinSIGHT_VSCode project root: python src/SVM.py
No sklearn SVM classifier is used. Thresholds are selected on validation only.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs'
OUT = DATA / 'svm_scratch'
PLOTS = OUT / 'graphs'
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

def save_show(name):
    plt.tight_layout()
    plt.savefig(PLOTS / (name + '.png'), dpi=140)
    plt.close()

def fit_svm_sgd(X, y, C=1.0, epochs=35, learning_rate=0.01, seed=42):
    """
    Linear soft-margin SVM trained from scratch with stochastic subgradient descent.
    Objective: (1/2)||w||^2 + C * sum(max(0, 1 - y_i(w.x_i+b))).
    y must be encoded as -1/+1. No sklearn SVM is used.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    # Normalize C's data term by n for stable optimization.
    lam = 1.0 / max(C * n, 1e-12)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        eta = learning_rate / np.sqrt(epoch)
        class_weights = {1: n / (2.0 * max(np.sum(y == 1), 1)),
                         -1: n / (2.0 * max(np.sum(y == -1), 1))}
        for i in order:
            margin = y[i] * (np.dot(w, X[i]) + b)
            # L2 regularization gradient plus class-balanced hinge-loss subgradient
            w *= (1.0 - eta * lam)
            if margin < 1.0:
                cw = class_weights[int(y[i])]
                w += eta * cw * y[i] * X[i]
                b += eta * cw * y[i]
    return {"w": w, "b": float(b), "C": C, "epochs": epochs}

def predict_svm(model, X):
    scores = X @ model["w"] + model["b"]
    return (scores >= 0).astype(int), scores

def metric_values(y_true, y_pred, scores=None):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)
    balanced_accuracy = (recall + specificity) / 2
    # AUC by pairwise rank/Mann-Whitney formula, no sklearn
    auc = float("nan")
    if scores is not None:
        pos, neg = scores[y_true == 1], scores[y_true == 0]
        if len(pos) and len(neg):
            # Rank scores; average ties
            all_s = np.concatenate([pos, neg])
            order = np.argsort(all_s, kind="mergesort")
            ranks = np.empty(len(all_s), dtype=float)
            i = 0
            while i < len(order):
                j = i + 1
                while j < len(order) and all_s[order[j]] == all_s[order[i]]:
                    j += 1
                ranks[order[i:j]] = (i + 1 + j) / 2.0
                i = j
            rank_sum = ranks[:len(pos)].sum()
            auc = float((rank_sum - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall,
            "Specificity": specificity, "F1-score": f1,
            "Balanced accuracy": balanced_accuracy, "ROC-AUC": auc,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}

def plot_confusion(metrics, title, filename):
    cm = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted stayed", "Predicted left"],
                yticklabels=["Actually stayed", "Actually left"])
    plt.title(title); plt.xlabel("Prediction"); plt.ylabel("Actual")
    save_show(filename)

def plot_roc(y, scores, title, filename):
    thresholds = np.unique(scores)
    # Include endpoints; evaluate thresholded score manually
    points = [(0.0, 0.0)]
    positives = max(np.sum(y == 1), 1); negatives = max(np.sum(y == 0), 1)
    for t in np.sort(thresholds)[::-1]:
        pred = scores >= t
        points.append((np.sum((y == 0) & pred)/negatives, np.sum((y == 1) & pred)/positives))
    points.append((1.0, 1.0))
    points = np.array(points)
    order = np.argsort(points[:, 0], kind="mergesort")
    pts = points[order]
    plt.figure(figsize=(7, 5))
    plt.plot(pts[:,0], pts[:,1], label="SVM ROC")
    plt.plot([0,1],[0,1],"--",label="Random baseline")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend()
    save_show(filename)

def main():
    # Load exactly the same 8,000 training and 2,000 testing customers as other models.
    needed = ['X_train_encoded.csv', 'X_test_encoded.csv', 'y_train.csv', 'y_test.csv']
    for name in needed:
        if not (DATA / name).exists():
            raise FileNotFoundError(f'Missing {name}. Run python src/FULL_EDA.py first.')
    x_train_df = pd.read_csv(DATA / 'X_train_encoded.csv')
    x_test_df = pd.read_csv(DATA / 'X_test_encoded.csv')
    y_train = pd.read_csv(DATA / 'y_train.csv').iloc[:, 0].to_numpy(dtype=int)
    y_test = pd.read_csv(DATA / 'y_test.csv').iloc[:, 0].to_numpy(dtype=int)
    x_test_df = x_test_df.reindex(columns=x_train_df.columns, fill_value=0)
    assert len(x_train_df) == len(y_train) == 8000
    assert len(x_test_df) == len(y_test) == 2000
    assert list(x_train_df.columns) == list(x_test_df.columns)
    print('Shared training rows:', len(y_train), 'Shared testing rows:', len(y_test))

    # Validation comes from training data only. Never use test rows to select settings.
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train_df, y_train, test_size=0.20, random_state=17, stratify=y_train)
    scaler_val = StandardScaler()
    x_fit_scaled = scaler_val.fit_transform(x_fit)
    x_val_scaled = scaler_val.transform(x_val)

    results = []
    thresholds = [0.0, -0.1, -0.2, -0.3, -0.4, 0.1, 0.2]
    for C in [0.1, 1.0, 10.0]:
        # Keep original from-scratch hinge-loss and SGD weight updates.
        model = fit_svm_sgd(x_fit_scaled, np.where(y_fit == 1, 1, -1),
                            C=C, epochs=28, learning_rate=0.01, seed=42)
        _, scores = predict_svm(model, x_val_scaled)
        for threshold in thresholds:
            predicted = (scores >= threshold).astype(int)
            values = metric_values(y_val, predicted, scores)
            results.append({'C': C, 'Threshold': threshold, **values})
        print('Validated C =', C)

    table = pd.DataFrame(results)
    table.to_csv(OUT / 'validation_results.csv', index=False)
    # Prioritize F1 and use Recall as a tie-breaker: avoids selecting all-positive results.
    best = table.sort_values(['F1-score', 'Recall'], ascending=[False, False]).iloc[0]
    best_C = float(best['C'])
    threshold = float(best['Threshold'])
    print('Selected C:', best_C, '| Threshold:', threshold)

    # Refit scaler and SVM on ALL training rows, then evaluate test ONCE.
    final_scaler = StandardScaler()
    x_full = final_scaler.fit_transform(x_train_df)
    x_test = final_scaler.transform(x_test_df)
    model = fit_svm_sgd(x_full, np.where(y_train == 1, 1, -1),
                        C=best_C, epochs=28, learning_rate=0.01, seed=42)
    _, scores = predict_svm(model, x_test)
    predictions = (scores >= threshold).astype(int)
    metrics = metric_values(y_test, predictions, scores)
    print('\nFINAL SVM TEST RESULTS')
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}' if isinstance(value, float) else f'{name}: {value}')
    pd.DataFrame([{'C': best_C, 'Threshold': threshold, **metrics}]).to_csv(OUT / 'test_metrics.csv', index=False)
    pd.DataFrame({'Actual': y_test, 'Predicted': predictions, 'DecisionScore': scores}).to_csv(OUT / 'predictions.csv', index=False)
    parameters = {
        'algorithm': 'Linear SVM using manually implemented hinge-loss SGD',
        'C': best_C, 'threshold': threshold, 'epochs': 28,
        'weights': model['w'].tolist(), 'bias': model['b'],
        'feature_names': x_train_df.columns.tolist(),
        'scaler_mean': final_scaler.mean_.tolist(),
        'scaler_scale': final_scaler.scale_.tolist(),
    }
    with open(OUT / 'model_parameters.json', 'w', encoding='utf-8') as file:
        json.dump(parameters, file, indent=2)
    plot_confusion(metrics, 'SVM Confusion Matrix', 'confusion_matrix')
    plot_roc(y_test, scores, 'SVM ROC Curve', 'roc_curve')
    print('Results saved to:', OUT)

if __name__ == '__main__':
    main()
