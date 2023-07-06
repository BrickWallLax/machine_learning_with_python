import math
import quandl
import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# -------------------------------------------------------------------------------------------------------------------- #
# Correctly classify relevant features to input into model #

df = quandl.get('WIKI/GOOGL')  # Get stock prices from quandl
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Create FEATURES based on the relationship between different information relating to what you
# are trying to predict
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close', 'Adj. Volume', 'HL_PCT', 'PCT_change']]  # Add the FEATURES into the dataframe
forecast_col = 'Adj. Close'  # Set up what you are trying to predict

df.fillna(-99999, inplace=True)  # Fill in missing data

# -------------------------------------------------------------------------------------------------------------------- #
# Set up forecast, move label, and set up X and y #

forecast_out = int(math.ceil(.1*len(df)))  # Calculates how many days to forecast out

# Creates a copy so that I can graph it
# df = df.iloc[:-forecast_out]
# df_graph = df.copy()

# Shifts each item in the forecast_col column up forcast_out days and sets it to the label column
df['label'] = df[forecast_col].shift(-forecast_out)

# Features = X ; Labels = y / .drop creates a new dataframe containing everything except the value provided
X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

X = preprocessing.scale(X)  # Preprocessing --> skip this step if you have a large amount of data
X_lately = X[-forecast_out:]  # X_lately is prediction data
X = X[:-forecast_out]  # X is data before predictions

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)  # Sets the 4 variables accordingly

# -------------------------------------------------------------------------------------------------------------------- #
# Set up the model, fit the training data and score based on the test data #

# clf = LinearRegression(n_jobs=-1) # n-jobs is amount of threads
# # clf = svm.SVR() --> Super easy to switch algorithms
# clf.fit(X_train, y_train)  # Training the model

# Pickle implementation:
# with open('linearregression.pickle', 'wb') as f: # Set up model file
#     pickle.dump(clf, f)
pickle_in = open('linearregression.pickle', 'rb') # Open model file
clf = pickle.load(pickle_in) # Load model file

# TIP: If you want to scale up, write your code on your own IDE then run a big server for only how long it takes to
# train and save the model, then scale down the server and use the model you created on your IDE

accuracy = clf.score(X_test, y_test)  # Gets an accuracy reading of the test values
forecast_set = clf.predict(X_lately)  # Predicts a value (or multiple values)
# print(forecast_set, accuracy, forecast_out)

# -------------------------------------------------------------------------------------------------------------------- #
# Connect forecast with accurate dates #

df['Forecast'] = np.nan  # Fills a column with nan (Not a Number) data
last_date = df.iloc[-1].name  # Gets the date of the last item in the dataframe
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]  # Replaces the nan values in the column

# -------------------------------------------------------------------------------------------------------------------- #
# Set up visual graph #

style.use('ggplot')  # Styling the plot
# df_graph['Adj. Close'].plot()
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.title('Google Stock')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
