import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
try:
    from IPython.display import display
except ImportError:
    display = print

print("Loading dataset...")
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

alphas_ridge = np.logspace(-4, 4, 100)

ridge_cv = RidgeCV(alphas=alphas_ridge, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_cv.predict(X_test_scaled)

alphas_lasso = np.logspace(-4, 4, 100)

lasso_cv = LassoCV(alphas=alphas_lasso, cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_cv.predict(X_test_scaled)

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MSE": mse, "R-squared": r2}

results = [
    evaluate_model(y_test, y_pred_lr, "Standard Linear Regression"),
    evaluate_model(y_test, y_pred_ridge, f"Ridge Regression (Best Alpha: {ridge_cv.alpha_:.4f})"),
    evaluate_model(y_test, y_pred_lasso, f"Lasso Regression (Best Alpha: {lasso_cv.alpha_:.4f})")
]
results_df = pd.DataFrame(results)
print("\n--- Performance Comparison ---")
display(results_df)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Standard Linear': lr.coef_,
    'Ridge': ridge_cv.coef_,
    'Lasso': lasso_cv.coef_
}).set_index('Feature')

coef_df.plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title('Comparison of Feature Coefficients (Shrinkage Effect)')
plt.ylabel('Coefficient Value')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend()
plt.tight_layout()
plt.show()

