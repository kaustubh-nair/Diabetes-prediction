import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import math
import seaborn as sns
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


data = pd.read_csv('./Pima_Indian_diabetes.csv')
data = data.dropna(axis=0)
main_data = data[['Glucose', 'Insulin']]

zero_insulin_data = main_data[ main_data.Insulin == 0 ]
print(zero_insulin_data.shape)
non_zero_insulin_data = main_data[ main_data.Insulin != 0]

train_X = non_zero_insulin_data['Glucose'].values.reshape(-1,1)
train_y = non_zero_insulin_data['Insulin'].values.reshape(-1,1)
val_X = zero_insulin_data['Glucose'].values.reshape(-1,1)

model = LinearRegression()
model.fit(train_X, train_y)
a = model.predict(val_X)
j = 0
print(main_data)
for i in main_data.index:
    if main_data.at[i, 'Insulin'] == 0:
        main_data.at[i, 'Insulin'] = a[j][0]
        j+=1
        if main_data.at[i, 'Insulin'] == 0:
            print(main_data.at[i, 'Insulin'])

print(main_data.sort_values(by=['Insulin'],ascending=True).head(10))
main_data[ main_data < 0 ] = 0
