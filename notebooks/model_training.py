import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv', header=1)
# print(dataset.head())
# print(dataset.info())

#Cleaning the dataset
# print(dataset[dataset.isnull().any(axis=1)])

#Now we know we have to regions in the dataset so we have to make a new column which defines the region
dataset.loc[:122, 'Region'] = 0
dataset.loc[122:, 'Region'] = 1


dataset[['Region']] = dataset[['Region']].astype(int)

#removing the null values
df = dataset.dropna().reset_index(drop=True)

df = df.drop(122).reset_index(drop=True)

#Now fix spaces in column names
df.columns = df.columns.str.strip()

#Changing the columns to as type integer
df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']] = df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']].astype(int)

#Changing the remaining dataset to float
df[['Rain','FFMC','DMC','DC','ISI','BUI','FWI']] = df[['Rain','FFMC','DMC','DC','ISI','BUI','FWI']].astype(float)
# print(df.info())
# print(df.head())

#Now lets save the cleaned dataset
df.to_csv('Algerian1_forest_fires_cleaned_dataset.csv', index=False)

#We are gonna perform exploratory data analysis
df_copy = df.drop(['day','month','year'], axis=1)


#For classes feature we have to perform one hot encoding
df_copy['Classes'] = np.where(df['Classes'].str.contains('not fire'),0,1)
# print(df_copy.head())

#Lets count the values of fire in classes
# print(df_copy['Classes'].value_counts())


#Now comes the visualisation part

plt.style.use('seaborn-v0_8-darkgrid')
df_copy.hist(bins=50, figsize=(20,15))
plt.show()

#Percentage for piechart
df_copy['Classes'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()


#Correlation and high colinearity
df_copy.corr()

#Monthy fire analysis
dftemp = df.loc[df['Region'] == 1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month', hue='Classes', data=dftemp)
plt.show()