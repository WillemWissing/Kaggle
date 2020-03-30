# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:14:15 2020

@author: Willem
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('train.csv')

#cleaning the text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

dataset = dataset.fillna('Unknown')
ps = PorterStemmer()
dataset['keyword'] = [ps.stem(word) for word in dataset['keyword'] if not word in set(stopwords.words('english'))]
keyword_sum = dataset.groupby(['keyword'])['target'].mean()

# dataset['location'] = dataset['location'].str.replace('[^a-zA-Z]', ' ',regex = True).str.lower()
# dataset['location'] = [ps.stem(word) for word in dataset['location'] if not word in set(stopwords.words('english'))]

# location_sum = dataset.groupby(['location'])['target'].sum()['target'].mean()
# location_sum = dataset.groupby(['location'])['target'].agg(['sum','mean'])


dataset['text'] = dataset['text'].str.replace('[^a-zA-Z]', ' ',regex = True)
dataset['text'] = dataset['text'].str.lower()
dataset['text'] = dataset['text'].str.split()

corpus = []

for i in range(0,len(dataset)):
    newtext = dataset['text'][i]
    newtext = [ps.stem(word) for word in newtext if not word in set(stopwords.words('english'))]
    newtext = ' '.join(newtext)
    corpus.append(newtext)
dataset['text'] = corpus

    
#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 800)

# X_location = cv.fit_transform(dataset['location']).toarray()
X_text = cv.fit_transform(dataset['text']).toarray()
X_keyword = cv.fit_transform(dataset['keyword']).toarray()
X = np.concatenate((X_text, X_keyword),axis=1)
y =  dataset.iloc[:,4].values


#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)

              
        
  
#Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#fitting logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

#predict the test set results
y_pred = classifier.predict(X_test)    

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

cm = confusion_matrix(y_test,y_pred)
report = classification_report(y_test, y_pred) 
scores = cross_val_score(classifier, X_train, y_train, cv = 5, scoring = 'f1')
f1_score_report = f1_score(y_test,y_pred)



