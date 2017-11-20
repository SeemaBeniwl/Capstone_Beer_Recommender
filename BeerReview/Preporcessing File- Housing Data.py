import pandas as pd
import numpy as np
train_data = pd.read_csv("clean_train.csv")
X_clean = train_data.iloc[:,[4]].values
y_clean = train_data.iloc[:,[80]].values

import matplotlib.pyplot as plt
plt.boxplot(y_clean)
"""
First try of removing outlier function
"""
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

y_rej = reject_outliers(y_clean)

plt.boxplot(X_clean)
X_rej = reject_outliers(X_clean)

X_df = pd.DataFrame(X_clean)
y_df = pd.DataFrame(y_clean)
train = pd.concat([X_df,y_df],axis=1)
train.columns = ['Area','Price']
X = train.iloc[0:1397,[0]].values
y = train.iloc[0:1397,[1]].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size = 1/3, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Housing Price ')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Housing Price ')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

"""
####################################################################################
"""

import pandas as pd
train_data = pd.read_csv("clean_train.csv")
train = train_data.iloc[:,[4,80]].values
train_df = pd.DataFrame(train, index= None)
train_df.columns = ['Area','Price']
train_df.isnull().values.any()

def replace(group):
    mean, std = group.mean(), group.std()
    outliers = (group - mean).abs() > 3*std
    group[outliers] = mean        # or "group[~outliers].mean()"
    return group

X = train_df.groupby('Area').transform(replace)
plt.boxplot(y['Area'])
plt.show()
y = train_df.groupby('Price').transform(replace)
plt.scatter(train_df['Area'], train_df['Price'], color = 'red')

rej = reject_outliers(train)

"""
##################################################################################
"""
import pandas as pd
train_data = pd.read_csv("clean_train.csv")
train = train_data.iloc[:,[4,80]].values
train_df = pd.DataFrame(train, index= None)
train_df.columns = ['Area','Price']
train_df.isnull().values.any()
# Removing Outliers
a = train_df[train_df.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
a.isnull().values.any()
# Again Removing outliers
b = a[a.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
b.isnull().values.any()
# Converting dataframe into csv file
b.to_csv("clean_train1.csv")


# Checking boxplot
plt.boxplot(train_df['Area'])
plt.boxplot(train_df['Price'])
# Setting data variables
clean_train1 = pd.read_csv("clean_train1.csv")
clean_train1 = clean_train1.drop('Unnamed: 0', 1)
X = clean_train1.iloc[:,[0]].values
y = clean_train1.iloc[:,[1]].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Housing Price ')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Housing Price ')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
err = mean_squared_error(y_test,y_pred)
r2_score(y_test, y_pred)
rmse = math.sqrt(err)
abserr = mean_absolute_error(y_test, y_pred)


plt.hist(X, normed=False, bins=40)



