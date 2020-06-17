import numpy as np
import pandas as pd
import pickle

dataset=pd.read_csv('diabetes.csv')

dataset_copy=dataset.copy(deep=True)
dataset_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=dataset_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


dataset_copy['Glucose'].fillna(dataset_copy['Glucose'].mean(), inplace=True)
dataset_copy['BloodPressure'].fillna(dataset_copy['BloodPressure'].mean(), inplace=True)
dataset_copy['SkinThickness'].fillna(dataset_copy['SkinThickness'].median(), inplace=True)
dataset_copy['Insulin'].fillna(dataset_copy['Insulin'].median(), inplace=True)
dataset_copy['BMI'].fillna(dataset_copy['BMI'].median(), inplace=True)


# Model Building
from sklearn.model_selection import train_test_split
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Using randomforest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

#creating a pickle file for classifier
filename = 'diabetes_pred.pkl'
pickle.dump(classifier, open(filename, 'wb'))