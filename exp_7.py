import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

train = fetch_20newsgroups(
    subset='train',
    remove=('headers', 'footers', 'quotes')
)

test = fetch_20newsgroups(
    subset='test',
    remove=('headers', 'footers', 'quotes')
)

vectorizer = CountVectorizer(
    stop_words='english', max_features=15000
)

X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)

y_train = train.target
y_test = test.target

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

pred1 = mnb.predict(X_test)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)

pred2 = bnb.predict(X_test)

print("Multinomial NB")
print("Accuracy:", accuracy_score(y_test, pred1))
print("F1 Score:", f1_score(y_test, pred1, average='macro'))

print("\nBernoulli NB")
print("Accuracy:", accuracy_score(y_test, pred2))
print("F1 Score:", f1_score(y_test, pred2, average='macro'))
