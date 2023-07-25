import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

style.use('ggplot')

df = pd.read_excel('titanic_data.xlsx', engine='openpyxl')
original_df = pd.DataFrame.copy(df)

df.apply(pd.to_numeric, errors='ignore')
df.drop(['body', 'name', 'boat', 'ticket'], axis=1, inplace=True)
df.fillna(0, inplace=True)


def handle_nonnumerical_data(df):
    columns = df.columns.values
    text_digit_vals = {}  # Initialize the dictionary outside the loop

    for column in columns:
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)

            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # Create a new key for each new value
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(lambda val: text_digit_vals[val], df[column]))

    return df


df = handle_nonnumerical_data(df)

X = np.array(df.drop(['survived'], axis=1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

df.drop(['sex'], axis=1, inplace=True)

# correct = 0
# for i in range(len(X)):
#     predict_me = np.array(X[i].astype(float))
#     predict_me.reshape(1, len(predict_me))
#     prediction = clf.predict([predict_me])
#     if prediction[0] == y[i]:
#         correct += 1
#
# print(correct/len(X))

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate
    print(original_df[(original_df['cluster_group'] == i)].describe())

print(survival_rates)