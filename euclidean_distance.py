from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import warnings

import pandas
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')

# -------------------------------------------------------------------------------------------------------------------- #
# def euclidean_distance(q, p):
#     total = 0
#     for i in range(len(q)):
#         total += (q[i] - p[i]) ** 2
#     return sqrt(total)

# dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
# new_features = [4, 4.5]
# -------------------------------------------------------------------------------------------------------------------- #


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than the total data groups')
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = euclidean_distance(features, predict)
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence


df = pandas.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
full_data = df.astype(float).values.tolist()

accuracies = []

for j in range(25):
    random.shuffle(full_data)

    # Set up training and testing data
    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size*len(full_data))] # 80% of the data
    test_data = full_data[-int(test_size*len(full_data)):] # 20% of the data

    # Remove the class column
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in train_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    # For all the data, run k_nearest_neighbors and tally votes, then find accuracy of the votes compared to actual results
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            # else:
            #     print(confidence)
            total += 1

    accuracy = correct/total
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))
# -------------------------------------------------------------------------------------------------------------------- #
# result = k_nearest_neighbors(dataset, new_features)
# print(result)
#
# # [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# # plt.scatter(new_features[0], new_features[1], s=200, color=result)
# # plt.show()
# -------------------------------------------------------------------------------------------------------------------- #



