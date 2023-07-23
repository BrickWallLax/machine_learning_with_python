import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
style.use('ggplot')

df = pd.read_excel('titanic_data.xlsx', engine='openpyxl')
df.drop(['body', 'name'], axis=1, inplace=True)
df.fillna(0, inplace=True)


def handle_nonnumerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[columns]))
    return df


df = handle_nonnumerical_data(df)
print(df.head())
