import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# -------------------------------------------------------------------------------------------------------------------- #
# Setting up data for training #
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)

df.drop(['id'], axis=1, inplace=True)

X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train, y_train)

# with open('knearestneighbors.pickle', 'wb') as f:  # Set up model file
#     pickle.dump(clf, f)
pickle_in = open('knearestneighbors.pickle', 'rb')  # Open model file
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

print(accuracy)

example_measures = np.array([[3, 9, 2, 7, 4, 10, 2, 7, 3], [3, 7, 2, 5, 4, 1, 2, 7, 3], [3, 7, 2, 5, 4, 1, 10, 7, 3]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)
