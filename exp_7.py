import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score
try:
    from IPython.display import display
except ImportError:
    display = print

print("Downloading/Loading 20 Newsgroups dataset (this may take a moment)...")

train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
X_train, y_train = train_data.data, train_data.target
X_test, y_test = test_data.data, test_data.target
print(f'Dataset loaded: {len(X_train)} training documents, {len(X_test)} test documents.')

vectorizer = CountVectorizer(stop_words='english', max_features=15000)
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_counts, y_train)
mnb_predictions = mnb.predict(X_test_counts)
mnb_metrics = {
    'Accuracy': accuracy_score(y_test, mnb_predictions),
    'F1-score (Macro)': f1_score(y_test, mnb_predictions, average='macro')
}

bnb = BernoulliNB(binarize=0.0)
bnb.fit(X_train_counts, y_train)
bnb_predictions = bnb.predict(X_test_counts)
bnb_metrics = {
    'Accuracy': accuracy_score(y_test, bnb_predictions),
    'F1-score (Macro)': f1_score(y_test, bnb_predictions, average='macro')
}

comparison_df = pd.DataFrame([mnb_metrics, bnb_metrics], index=['Multinomial NB', 'Bernoulli NB'])
print("\n--- Model Performance Comparison ---")
display(comparison_df)

