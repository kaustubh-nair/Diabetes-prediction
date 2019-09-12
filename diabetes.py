#0.7218934911242604
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
import math
import seaborn as sns
import numpy as np
import seaborn as sns

data = pd.read_csv('./Pima_Indian_diabetes.csv')
data[ data < 0 ] = np.NaN
data['Insulin'] = data['Insulin'].map(lambda x: 80 if x >250 else x)
data = data.dropna(axis=0)
sns.distplot(a=data['Pregnancies'], kde=False)
features = ['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction','Age']
X = data[features]
y = data['Outcome']
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)

print(train_X.describe())

model = LogisticRegression(random_state=1)
predictions = cross_val_score(model, train_X, train_y, cv=5, scoring='accuracy')
print(predictions)
print(predictions.mean())
