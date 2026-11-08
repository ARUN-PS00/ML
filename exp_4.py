import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
try:
    from IPython.display import display
except ImportError:
    display = print

print("Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mle_model = LogisticRegression(C=np.inf, max_iter=1000, random_state=42)
mle_model.fit(X_train_scaled, y_train)

map_l2_model = LogisticRegression(l1_ratio=0, C=0.1, max_iter=1000, random_state=42)
map_l2_model.fit(X_train_scaled, y_train)

map_l1_model = LogisticRegression(l1_ratio=1, solver='saga', C=0.1, max_iter=10000, random_state=42)
map_l1_model.fit(X_train_scaled, y_train)

def get_metrics(model, X_t, y_t, X_v, y_v, name):
    train_pred = model.predict(X_t)
    test_pred = model.predict(X_v)
    train_prob = model.predict_proba(X_t)
    test_prob = model.predict_proba(X_v)
    return {
        "Estimation Framework": name,
        "Train Accuracy": accuracy_score(y_t, train_pred),
        "Test Accuracy": accuracy_score(y_v, test_pred),
        "Train Log-Loss": log_loss(y_t, train_prob),
        "Test Log-Loss": log_loss(y_v, test_prob)
    }

results = [
    get_metrics(mle_model, X_train_scaled, y_train, X_test_scaled, y_test, "MLE(No Regularization)"),
    get_metrics(map_l2_model, X_train_scaled, y_train, X_test_scaled, y_test, "MAP L2 (Gaussian Prior), C=0.1"),
    get_metrics(map_l1_model, X_train_scaled, y_train, X_test_scaled, y_test, "MAP L1 (Laplace Prior, C=0.1)")
]

results_df = pd.DataFrame(results)
print("\n--- Model Performance Comparison ---")
display(results_df)

coef_df = pd.DataFrame({
    'Feature': data.feature_names,
    'MLE (Unregularized)': mle_model.coef_[0],
    'MAP L2 (Ridge)': map_l2_model.coef_[0],
    'MAP L1 (Lasso)': map_l1_model.coef_[0]
}).set_index('Feature')

print("\n--- Parameter Estimates Comparison ---")
display(coef_df.head())

plt.figure(figsize=(14, 7))
x_indices = np.arange(len(coef_df))
bar_width = 0.25
plt.bar(x_indices - bar_width, coef_df['MLE (Unregularized)'], width=bar_width, label='MLE (No Penalty)', color='crimson')
plt.bar(x_indices, coef_df['MAP L2 (Ridge)'], width=bar_width, label='MAP L2 (Gaussian Prior)', color='royalblue')
plt.bar(x_indices + bar_width, coef_df['MAP L1 (Lasso)'], width=bar_width, label='MAP L1 (Laplace Prior)', color='forestgreen')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Parameter Estimate Comparisons: MLE vs. MAP (L1 & L2)', fontsize=14)
plt.xlabel('Feature Index', fontsize=12)
plt.ylabel('Coefficient Weight value', fontsize=12)
plt.xticks(x_indices, range(len(coef_df)))
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

