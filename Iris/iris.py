# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:48:15 2020

@author: Willem
"""

#importing libraries
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dataset =pd.read_csv('iris.csv')

Features = dataset.drop(dataset.columns[5], axis=1)
Target = dataset.iloc[:,5]

x_train, x_test, y_train, y_test = train_test_split(Features,Target,test_size = 0.5)

clf = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state = 1)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accs = accuracy_score(y_test,y_pred)
confx = confusion_matrix(y_test,y_pred)
clfrep = classification_report(y_test,y_pred)