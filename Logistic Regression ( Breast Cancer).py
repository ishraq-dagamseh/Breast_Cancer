# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:00:25 2021

@author:Ishraq 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
#1. Read data
dataset=pd.read_csv('data.csv')
#dataframe = pd.DataFrame(dataset)

# 2.decided the features & labels
X = dataset.iloc[:, 2:-1].values
y=dataset.iloc[:,1].values

# label encoder just to y

from sklearn.preprocessing import LabelEncoder


#encoding the dependent variables
labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y)




#splitting X&y into training & testing
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)  

# must rescale data befor it (y-test & y-pred)

#feature scaling from x
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
X_train=SC.fit_transform(X_train)
X_test=SC.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

Classifier= LogisticRegression(random_state=0)

Classifier.fit(X_train,y_train)


y_pred=Classifier.predict(X_test)

# evalute using CM

from sklearn.metrics import confusion_matrix,classification_report


CM_LR = confusion_matrix(y_test,y_pred)
print(CM_LR)

print(classification_report(y_test, y_pred, target_names=['M','B']))

import seaborn as sns
sns.heatmap(CM_LR , annot=True)
