# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hhu9giWu7OO8IcCQosoSRft69gc6e_uu

## ML functions (internal cross-valitation)
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder

 
def my_inner_cv(X,y,model,cv,param_grid,test_size,random_state):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state,stratify=y)

  grid_search = GridSearchCV(model, param_grid=param_grid,cv=cv) 
  grid_search.fit(X_train, y_train)

  accuracy=accuracy_score(y_test,grid_search.best_estimator_.predict(X_test))
  precision=precision_score(y_test,grid_search.best_estimator_.predict(X_test),average='weighted')
  
  return([grid_search.best_params_,accuracy,precision,])

  #print('Best hyper-parameters:',grid_search.best_params_)
  #print('Average accuracy, precision= ', accuracy,precision)     


def my_logistic_regression(X,y,cv=5,param_grid={},test_size=0.2,random_state=101):
  model=LogisticRegression(solver='newton-cg',multi_class='ovr',penalty='l2')
  result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state)
  return(result)

def my_decision_tree(X,y,cv=5,param_grid={},test_size=0.2,random_state=101):
  model=DecisionTreeClassifier()
  result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state)
  return(result)

def my_random_forest(X,y,cv=5,param_grid={},n_jobs=-1,test_size=0.2,random_state=101):
  model=RandomForestClassifier(n_jobs=n_jobs)
  result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state)
  return(result)

def my_boosting(X,y,cv=5,param_grid={},test_size=0.2,random_state=101):
  model=GradientBoostingClassifier()
  result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state)
  return(result)

def my_svm(X,y,cv=5,param_grid={},test_size=0.2,random_state=101):
  #==============
  import warnings
  warnings.simplefilter(action='ignore', category=FutureWarning) #avoid the "future warning" that I think is not important"
  #==============
  model=SVC()
  result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state)
  return(result)