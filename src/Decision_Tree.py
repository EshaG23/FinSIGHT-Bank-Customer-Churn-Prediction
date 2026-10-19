# FinSIGHT: simple decision trees made from NumPy, not sklearn classifiers.
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'

class SimpleTree:
    def __init__(self, max_depth=6, min_leaf=15, max_features=None, seed=42):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.random = np.random.default_rng(seed)

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)
        self.n_features = self.X.shape[1]
        self.root = self.grow(np.arange(len(y)), 0)
        del self.X, self.y
        return self

    def grow(self, rows, depth):
        y = self.y[rows]
        probability = float(np.mean(y))
        node = {'prob': probability}
        if depth >= self.max_depth or len(rows) < 2 * self.min_leaf or probability in (0.0, 1.0):
            return node
        features = np.arange(self.n_features)
        if self.max_features is not None:
            features = self.random.choice(features, min(self.max_features, len(features)), replace=False)
        best_gain, best_feature, best_cut = -1.0, None, None
        parent_impurity = 2 * probability * (1 - probability)
        for feature in features:
            values = self.X[rows, feature]
            order = np.argsort(values, kind='mergesort')
            sorted_x = values[order]
            sorted_y = y[order]
            positions = np.arange(self.min_leaf, len(rows) - self.min_leaf + 1)
            positions = positions[sorted_x[positions - 1] < sorted_x[positions]]
            if len(positions) == 0:
                continue
            left_pos = np.cumsum(sorted_y)[positions - 1]
            total_pos = np.sum(sorted_y)
            left_n = positions
            right_n = len(rows) - positions
            left_p = left_pos / left_n
            right_p = (total_pos - left_pos) / right_n
            impurity = (left_n * (2*left_p*(1-left_p)) + right_n * (2*right_p*(1-right_p))) / len(rows)
            idx = int(np.argmin(impurity))
            gain = parent_impurity - impurity[idx]
            if gain > best_gain:
                best_gain, best_feature = float(gain), int(feature)
                pos = positions[idx]
                best_cut = float((sorted_x[pos-1] + sorted_x[pos]) / 2)
        if best_feature is None or best_gain <= 1e-10:
            return node
        left = rows[self.X[rows, best_feature] <= best_cut]
        right = rows[self.X[rows, best_feature] > best_cut]
        node.update({'feature': best_feature, 'cut': best_cut,
                     'left': self.grow(left, depth + 1),
                     'right': self.grow(right, depth + 1)})
        return node

    def predict_probability(self, X):
        X = np.asarray(X, dtype=float)
        result = np.zeros(len(X))
        for i, row in enumerate(X):
            node = self.root
            while 'feature' in node:
                node = node['left'] if row[node['feature']] <= node['cut'] else node['right']
            result[i] = node['prob']
        return result


def evaluate(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0,1]).ravel()
    return {'Accuracy': accuracy_score(y, predictions),
            'Precision': precision_score(y, predictions, zero_division=0),
            'Recall': recall_score(y, predictions, zero_division=0),
            'F1': f1_score(y, predictions, zero_division=0),
            'ROC_AUC': roc_auc_score(y, probabilities),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}


def choose_threshold(y, probabilities):
    # Prefer F1, which rewards recall without ignoring false alarms.
    rows = []
    for threshold in np.round(np.arange(0.20, 0.601, 0.025), 3):
        result = evaluate(y, probabilities, threshold)
        rows.append({'Threshold': threshold, **result})
    table = pd.DataFrame(rows)
    best = table.sort_values(['F1', 'Precision'], ascending=False).iloc[0]
    return float(best['Threshold']), table


def get_data():
    needed = ['X_train_encoded.csv', 'X_test_encoded.csv', 'y_train.csv', 'y_test.csv']
    for file in needed:
        if not (OUT / file).exists():
            raise FileNotFoundError(f'Missing {OUT / file}. Run FULL_EDA.py first.')
    train = pd.read_csv(OUT / needed[0]); test = pd.read_csv(OUT / needed[1])
    y_train = pd.read_csv(OUT / needed[2]).iloc[:, 0].to_numpy(dtype=int)
    y_test = pd.read_csv(OUT / needed[3]).iloc[:, 0].to_numpy(dtype=int)
    assert train.columns.tolist() == test.columns.tolist()
    return train.to_numpy(dtype=float), test.to_numpy(dtype=float), y_train, y_test, train.columns.tolist()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, names = get_data()
    # Split only the original training portion for tuning. Do not touch the final test set.
    xa, xv, ya, yv = train_test_split(X_train, y_train, test_size=0.25,
                                      random_state=42, stratify=y_train)
    candidates = [(5, 10), (7, 15), (9, 25)]
    validation_rows=[]
    best_score=-1
    for config in candidates:
        print('Testing', config, flush=True)
        model = SimpleTree(config[0], config[1])
        model.fit(xa, ya)
        scores = model.predict_probability(xv)
        threshold, thresholds = choose_threshold(yv, scores)
        result = evaluate(yv, scores, threshold)
        validation_rows.append({'Configuration': str(config), 'Threshold': threshold, **result})
        if result['F1'] > best_score:
            best_score=result['F1']; selected=config; selected_threshold=threshold
    # Refit the chosen model on all 8,000 training examples.
    final_model = SimpleTree(selected[0], selected[1])
    print('Refitting final model:', selected, flush=True)
    final_model.fit(X_train, y_train)
    test_scores=final_model.predict_probability(X_test)
    metrics=evaluate(y_test, test_scores, selected_threshold)
    folder=OUT / 'decision_tree_scratch'
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(folder/'validation_results.csv',index=False)
    pd.DataFrame([{'Configuration':str(selected), 'Threshold':selected_threshold, **metrics}]).to_csv(folder/'test_metrics.csv',index=False)
    pd.DataFrame({'y_true':y_test,'y_pred':(test_scores>=selected_threshold).astype(int),
                  'y_score':test_scores}).to_csv(folder/'predictions.csv',index=False)
    # Save the real trained model and exact feature order for the future Streamlit app.
    with open(folder/'model.pkl','wb') as file:
        pickle.dump({'tree_roots': [tree.root for tree in final_model.trees] if hasattr(final_model, 'trees') else [final_model.root], 'feature_columns':names,'threshold':selected_threshold, 'model_type':'Decision_Tree'},file)
    print('\nSelected:', selected, 'Threshold:',selected_threshold)
    print('Final test results:')
    print(pd.Series(metrics).to_string())
    print('Saved to:',folder)
