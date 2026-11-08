import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
try:
    from IPython.display import display
except ImportError:
    display = print

print("Loading 20 Newsgroups dataset (subset to 4 categories for speed)...")
categories = ['sci.space', 'talk.religion.misc', 'comp.graphics', 'rec.autos']
train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(train_data.data)
X_test = vectorizer.transform(test_data.data)
y_train = train_data.target
y_test = test_data.target
vocab = vectorizer.get_feature_names_out()

mle_model = MultinomialNB(alpha=1e-10, force_alpha=True)
mle_model.fit(X_train, y_train)

map_weak_model = MultinomialNB(alpha=0.1)
map_weak_model.fit(X_train, y_train)

map_strong_model = MultinomialNB(alpha=1.0)
map_strong_model.fit(X_train, y_train)

map_vstrong_model = MultinomialNB(alpha=5.0)
map_vstrong_model.fit(X_train, y_train)

def evaluate_model(model, name, alpha_val):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return {"Estimation Framework": name, "Prior (Alpha)": alpha_val, "Train Accuracy": train_acc, "Test Accuracy": test_acc}

results = [
    evaluate_model(mle_model, "MLE", "1e-10 (Effectively 0)"),
    evaluate_model(map_weak_model, "MAP (Weak Prior)", "0.1"),
    evaluate_model(map_strong_model, "MAP (Strong Prior)", "1.0"),
    evaluate_model(map_vstrong_model, "MAP (Very Strong)", "5.0")
]
results_df = pd.DataFrame(results)
print("\n--- Model Performance Comparison ---")
display(results_df)

class_idx = categories.index('sci.space')
mle_probs = np.exp(mle_model.feature_log_prob_[class_idx])
map_strong_probs = np.exp(map_strong_model.feature_log_prob_[class_idx])
word_counts = X_train[y_train == class_idx].sum(axis=0).A1
rare_word_idx = np.where((word_counts > 0) & (word_counts < 3))[0][0]
frequent_word_idx = np.argsort(word_counts)[-1]
rare_word = vocab[rare_word_idx]
freq_word = vocab[frequent_word_idx]
print("\n--- Parameter Estimation Comparison (Probability of Word given 'sci.space') ---")
param_df = pd.DataFrame({
    'Word Type': [f"Rare Word ('{rare_word}')", f"Frequent Word ('{freq_word}')"],
    'Training Count in Class': [word_counts[rare_word_idx], word_counts[frequent_word_idx]],
    'MLE Probability': [mle_probs[rare_word_idx], mle_probs[frequent_word_idx]],
    'MAP (Alpha=1.0) Probability': [map_strong_probs[rare_word_idx], map_strong_probs[frequent_word_idx]]
})
display(param_df)

alphas = np.logspace(-3, 1, 50)
test_accuracies = []
for a in alphas:
    clf = MultinomialNB(alpha=a)
    clf.fit(X_train, y_train)
    test_accuracies.append(accuracy_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(12, 8))
plt.plot(alphas, test_accuracies, marker='.', color='royalblue')
plt.xscale('log')
plt.title('Test Accuracy vs. Dirichlet Prior Strength ($\\alpha$)', fontsize=14)
plt.xlabel('Prior Strength / Smoothing Parameter ($\\alpha$)', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='Standard Laplace ($\\alpha=1.0$)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

