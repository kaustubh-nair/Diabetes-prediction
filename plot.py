import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
import math
import seaborn as sns
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


data = pd.read_csv('./Pima_Indian_diabetes.csv')
data[ data < 0 ] = np.NaN
#data['Insulin'] = data['Insulin'].map(lambda x: 80 if x >250 else x)
data = data.dropna(axis=0)
features = ['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction','Age']
features1 = ['Pregnancies','Glucose','Insulin', 'BloodPressure', 'Outcome']
features2 = ['SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction','Age','Outcome']
#for feature in features:
    #plt.figure()
    #sns.distplot(a=data[feature], kde=False)
X = data[features]
y = data['Outcome']
plt.style.use('ggplot')
# Load some data
df = pd.DataFrame(data[features1], columns=features1)
scatter_matrix(df, alpha=0.2, figsize=(10, 10))
plt.show()
df = pd.DataFrame(data[features2], columns=features2)
scatter_matrix(df, alpha=0.2, figsize=(10, 10))
plt.show()



train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)


model = LogisticRegression(random_state=1)
predictions = cross_val_score(model, train_X, train_y, cv=5, scoring='accuracy')
