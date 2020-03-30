
"""
Created on Sun Oct  7 17:31:46 2018

@author: willem
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer  as Imputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


=======
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
>>>>>>> bb2b07ddac8e1041a0c5c1b7850a42ed63399ca3
#importing data
trainset,testset = train_test_split(pd.read_csv('train.csv'))

# testset = pd.read_csv('test.csv')
truetest = testset.iloc[:,1]
testset =testset.drop(columns=['Survived'])


X = trainset.drop(columns = ['Survived'])
Y = trainset.iloc[:,1]
    
def preprocessdata(dataset):
    categorical = pd.get_dummies(data=dataset,columns =['Embarked','Sex'])

    names= pd.DataFrame({"Names":dataset['Name']})


    column_titles = ['Mr.','Mrs.','Miss.','Master.', 'Rev.', 'Dr.','Col.','Mme.','Major.','Ms.','Lady.','Sir.', 'Mlle.','Capt.']

    names.reindex(columns = column_titles, fill_value=0 )

    titles = ['Mr\.','Mrs\.','Miss\.','Master\.', 'Rev\.', 'Dr\.','Col\.','Mme\.','Major\.','Ms\.','Lady\.','Sir\.', 'Mlle\.','Capt\.']

    for ColName, title in zip(column_titles, titles):
        names[ColName] = names['Names'].str.contains(title)

    names = names.drop(columns=['Names'])

    dataset = pd.concat([categorical, names],axis = 1)

    dataset = dataset.drop(columns = ['Name','Cabin','Ticket','PassengerId'])



    #taking care of missing data
    imputer = Imputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(dataset)
    dataset = imputer.transform(dataset)

  
    
    #Feature Scaling
    sc_X = StandardScaler()
    dataset= sc_X.fit_transform(dataset)
    
    return dataset

X = preprocessdata(X)    
predictset = preprocessdata(testset)
    
#random forest classifier

<<<<<<< HEAD
clf = RandomForestClassifier(n_estimators = 1000, max_depth = 5, random_state = 1)


param_grid = {
    'n_estimators': [200,500,700,1000, 1200],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [4,5,6,7, 8],
    'random_state': [1]}

CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid,cv=5)
CV_clf.fit(X,Y)
print(CV_clf.best_params_)

# clf = RandomForestClassifier(n_estimators = 200, max_depth = 7, max_features = 'auto')
# clf.fit(X,Y)

# clf_predict = clf.predict(predictset)    


# confusx_tree = confusion_matrix(truetest,clf_predict)
# acc_score_tree = accuracy_score(truetest,clf_predict)
# rep_tree = classification_report(truetest,clf_predict)

=======
clf = RandomForestClassifier(n_estimators = 1000, max_depth = 6, random_state = 1,)

# clf.fit(X,Y)

sel = SelectFromModel(clf)

clf.fit(X,Y)
sel.fit(X,Y)
sel.get_support()

selected_feat = pd.DataFrame(X).columns[(sel.get_support())]

    
clf_predict = clf.predict(predictset)    

crossval_clf = cross_val_score(estimator = clf, X = X, y = Y, cv = 10).mean()
acc_scores = accuracy_score(truetest,clf_predict)
    


clf.fit(X[:,sel.get_support()],Y)


clf_predict2 = clf.predict(predictset[:,sel.get_support()])    

crossval_clf2 = cross_val_score(estimator = clf, X = X[:,sel.get_support()], y = Y, cv = 10).mean()
acc_scores2 = accuracy_score(truetest,clf_predict2)



# confusx_tree = confusion_matrix(truetest,clf_predict)
# acc_score_tree = accuracy_score(truetest,clf_predict)
# rep_tree = classification_report(truetest,clf_predict)

# # SVM
# svm_clf = SVC()
# svm_clf.fit(X,Y)

# svm_predict = svm_clf.predict(predictset)

# crossval_svm = cross_val_score(estimator = svm_clf, X = X, y = Y, cv = 10).std()

# confusx_svm = confusion_matrix(truetest,svm_predict)
# acc_sccore_svm = accuracy_score(truetest,svm_predict)
# rep_svm = classification_report(truetest,svm_predict)


# #gaussianNB
# gnb_clf = GaussianNB()
# gnb_clf.fit(X,Y)

# gnb_predict = gnb_clf.predict(predictset)

# crossval_gnb = cross_val_score(estimator = gnb_clf, X = X, y = Y, cv = 10).std()

# confusx_gnb = confusion_matrix(truetest,gnb_predict)
# acc_sccore_gnb = accuracy_score(truetest,gnb_predict)
# rep_gnb = classification_report(truetest,gnb_predict)


# #K nearest neighbors
# knc_clf = KNeighborsClassifier()
# knc_clf.fit(X,Y)

# knc_predict = knc_clf.predict(predictset)

# crossval_knc = cross_val_score(estimator = knc_clf, X = X, y = Y, cv = 10).std()

# confusx_knc = confusion_matrix(truetest,knc_predict)
# acc_sccore_knc = accuracy_score(truetest,knc_predict)
# rep_knc = classification_report(truetest,knc_predict)


# #GradientBoost
# gb_clf = GradientBoostingClassifier()

# gb_clf.fit(X,Y)

# gb_predict = gb_clf.predict(predictset)

# crossval_gb = cross_val_score(estimator = gb_clf, X = X, y = Y, cv = 10).std()


# confusx_gb = confusion_matrix(truetest,gb_predict)
# acc_sccore_gb = accuracy_score(truetest,gb_predict)
# rep_gb = classification_report(truetest,gb_predict)
>>>>>>> bb2b07ddac8e1041a0c5c1b7850a42ed63399ca3


#output = pd.DataFrame({'PassengerId':testset.PassengerId, 'Survived' : predict})
# output.to_csv('2nd_submission.csv', index=False)
# print('Submission saved')
# print(output.shape)