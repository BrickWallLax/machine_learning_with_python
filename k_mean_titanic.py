import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
style.use('ggplot')

df = pd.read_excel('titanic_data.xlsx', engine='openpyxl')
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

clf = KMeans(n_clusters=2)
clf.fit(X)

df.drop(['sex'], axis=1, inplace=True)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me.reshape(1, len(predict_me))
    prediction = clf.predict([predict_me])
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))



