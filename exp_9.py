import pandas as pd
import numpy as np
from pprint import pprint

print("Downloading and processing Online Retail Dataset subset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

df = pd.read_excel(url, nrows=50000)

df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
df['TotalSum'] = df['Quantity'] * df['UnitPrice']
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalSum': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

rfm['Recency_Cat'] = pd.qcut(rfm['Recency'], q=3, labels=['Recent', 'Moderate', 'Old'])
rfm['Frequency_Cat'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=3, labels=['Low', 'Medium', 'High'])
rfm['Target_Segment'] = pd.qcut(rfm['Monetary'], q=2, labels=['Low_Value', 'High_Value'])

dataset = rfm[['Recency_Cat', 'Frequency_Cat', 'Target_Segment']].astype(str)
features = ['Recency_Cat', 'Frequency_Cat']
target = 'Target_Segment'

def calculate_entropy(target_column):
    elements, counts = np.unique(target_column, return_counts=True)
    entropy = np.sum([
        (-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
        for i in range(len(elements))
    ])
    return entropy

def information_gain(data, split_attribute, target_name):
    total_entropy = calculate_entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute], return_counts=True)

    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * calculate_entropy(data.where(data[split_attribute] == vals[i]).dropna()[target_name])
        for i in range(len(vals))
    ])
    return total_entropy - weighted_entropy

def id3(data, original_data, feature_list, target_name):
    if len(np.unique(data[target_name])) <= 1:
        return np.unique(data[target_name])[0]
    elif len(data) == 0:
        mode_val = np.unique(original_data[target_name])[
            np.argmax(np.unique(original_data[target_name], return_counts=True)[1])
        ]
        return mode_val
    elif len(feature_list) == 0:
        mode_val = np.unique(data[target_name])[
            np.argmax(np.unique(data[target_name], return_counts=True)[1])
        ]
        return mode_val
    else:
        gains = [information_gain(data, f, target_name) for f in feature_list]
        best_feature_idx = np.argmax(gains)
        best_feature = feature_list[best_feature_idx]

        tree = {best_feature: {}}
        remaining_features = [f for f in feature_list if f != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = id3(sub_data, original_data, remaining_features, target_name)
            tree[best_feature][value] = subtree
        return tree

print("\n--- Feature Information Gains at Root ---")
for feature in features:
    gain = information_gain(dataset, feature, target)
    print(f"{feature}: {gain:.4f}")

print("\n--- ID3 Decision Tree Structure ---")
decision_tree = id3(dataset, dataset, features, target)
pprint(decision_tree, indent=4)

