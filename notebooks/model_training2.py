import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Algerian1_forest_fires_cleaned_dataset.csv')
# print(df.head())

#Now we will only consider useful columns
df.drop(['day', 'month', 'year'] , axis=1, inplace=True)
# print(df.head())

df['Classes'] = np.where(df['Classes'].str.contains('not fire'),0,1)
print(df['Classes'].value_counts())

#Classifying the independant and dependant feature
x = df.drop('FWI', axis=1)
y = df['FWI']
# print(x.head())

#Now we will be performing train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2, random_state=42)


#Feature selection based on correlation
# print(x_train.corr())

#Checking for multicollinearity
# plt.figure(figsize=(12,10))
# corr = x_train.corr()
# sns.heatmap(corr, annot=True)
# plt.show()


#Now we will make a function to apply threshold to the collinearity
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_features = correlation(x_train, 0.85)
# print(corr_features)

#dropping the features based on correlation
x_train.drop(corr_features, axis=1, inplace=True)
x_test.drop(corr_features, axis=1, inplace=True)

#feature scaling and standardization
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)
# print(x_train_scaled)

#visualising the outliers using boxplot

# plt.subplots(figsize=(13,6))
# plt.subplot(1,2,1)
# sns.boxplot(data=x_train)
# plt.title("Before Scaling")

# plt.subplot(1,2,2)
# sns.boxplot(data=x_train_scaled)
# plt.title("After Scaling")
# plt.show()


#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
linreg = LinearRegression()
linreg.fit(x_train_scaled, y_train)
y_pred = linreg.predict(x_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(y_pred)
print(mse)
print(r2)


#Cross validation Lasso
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, random_state=0)
lasso.fit(x_train_scaled, y_train)
y_pred = lasso.predict(x_test_scaled)
plt.scatter(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mae)
print(mse)
print(r2)

#Cross validation using ridge
from sklearn.linear_model import RidgeCV
ridgecv = RidgeCV(cv=5)
ridgecv.fit(x_train_scaled, y_train)
y_pred = ridgecv.predict(x_test_scaled)
plt.scatter(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mae)
print(mse)
print(r2)


import pickle

pickle.dump(lasso, open('model.pkl', 'wb'))
pickle.dump(sc, open('scaling.pkl', 'wb'))
pickle.dump(ridgecv, open('model1.pkl', 'wb'))
