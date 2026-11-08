import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

architectures = {
    "Shallow & Small": (10,),
    "Shallow & Wide": (100,),
    "Deep & Funnel": (64, 32, 16),
    "Overly Complex": (128, 128, 64, 32)
}
results = {}

print("Training models...\n")
for name, hidden_layers in architectures.items():
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation="relu", max_iter=2000, random_state=42)
    start_time = time.time()
    mlp.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "Accuracy": acc,
        "Time (s)": train_time,
        "Epochs": mlp.n_iter_
    }

print("=== ARCHITECTURE COMPARISON ===")
for name, metrics in results.items():
    print(f"Architecture: {name}")
    print(f" Hidden Layers: {architectures[name]}")
    print(f" Accuracy: {metrics['Accuracy'] * 100:.2f}%")
    print(f" Train Time: {metrics['Time (s)']:.2f}s")
    print(f" Epochs to fit: {metrics['Epochs']}\n")

