# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:26:06 2021

@author: Ishraq 
"""
# #م = خبيث، ب = حميد(M = malignant, B = benign) 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
#1. Read data
dataset=pd.read_csv('data.csv')
#dataframe = pd.DataFrame(dataset)

# 2.decided the features & labels
X = dataset.iloc[:, 2:-1].values
y=dataset.iloc[:,1].values


#2.handling missing values from  
from sklearn.impute import SimpleImputer

# Mark the missing values
missing_values = dataset.isna().sum().reset_index()
print(missing_values)

# we noticed that no missing values 

# label encoder just to y
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


#encoding the dependent variables
labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y)


#splitting X&y into training & testing
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)  

# build model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
#  prediction 
y_pred=regressor.predict(X_test)


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_pred)
print('r2 socre is:' ,score)
print('mean_sqrd_error is==',mean_squared_error(y_test,y_pred))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_pred)))



# Backword Elimination on MLR( to delet non-important data from dataset)
# to defin constant to Multi linear regression model ( in dataset) bu adding ones column in position 0
X=np.append(np.ones((569,1)).astype(int),X ,axis=1)

# to fit all values with prediction 
import statsmodels.api as sm# to comput the p-value to all values
X_opt=np.array(X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 


# without x9 or symmetry_mean feature 
X_opt=np.array(X[:,[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x5 or smoothness_mean
X_opt=np.array(X[:,[0,1,2,3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x10 or col12()
X_opt=np.array(X[:,[0,1,2,3,4,6,7,8,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x16 or col19()
X_opt=np.array(X[:,[0,1,2,3,4,6,7,8,10,11,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x10 or col13()
X_opt=np.array(X[:,[0,1,2,3,4,6,7,8,10,11,14,15,17,18,20,21,22,23,24,25,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x12 or col17() or concave points_se feature 
X_opt=np.array(X[:,[0,1,2,3,4,6,7,8,10,11,14,15,18,20,21,22,23,24,25,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x12 or col18() or concave points_se feature 
X_opt=np.array(X[:,[0,1,2,3,4,6,7,8,10,11,14,15,20,21,22,23,24,25,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x4 or col4() or area_mean feature
X_opt=np.array(X[:,[0,1,2,3,6,7,8,10,11,14,15,20,21,22,23,24,25,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x5 or col7() 
X_opt=np.array(X[:,[0,1,2,3,6,8,10,11,14,15,20,21,22,23,24,25,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 


#without x8 or col14() 
X_opt=np.array(X[:,[0,1,2,3,6,8,10,11,15,20,21,22,23,24,25,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x14 or col25() 
X_opt=np.array(X[:,[0,1,2,3,6,8,10,11,15,20,21,22,23,24,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 


#without x2 or col2() 
X_opt=np.array(X[:,[0,1,3,6,8,10,11,15,20,21,22,23,24,26,27,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x14 or col27() 
X_opt=np.array(X[:,[0,1,3,6,8,10,11,15,20,21,22,23,24,26,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x8 or col20() 
X_opt=np.array(X[:,[0,1,3,6,8,10,11,15,21,22,23,24,26,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x10 or col23() 
X_opt=np.array(X[:,[0,1,3,6,8,10,11,15,21,22,24,26,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x2 or col3() 
X_opt=np.array(X[:,[0,1,6,8,10,11,15,21,22,24,26,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 
#without x4 or col10() 
X_opt=np.array(X[:,[0,1,6,8,11,15,21,22,24,26,28,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary

#without x10 or col28() 
X_opt=np.array(X[:,[0,1,6,8,11,15,21,22,24,26,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary 

#without x4 or col11() 
X_opt=np.array(X[:,[0,1,6,8,15,21,22,24,26,29]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_ols.summary())# to show the summary



##re- Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt, y, test_size=0.2, random_state=0)

y_predX_opt=regressor_ols.predict(X_test_opt)


# print("Mean squared error: %.2f" % np.mean((regressor.predict(X_test) - y_test) ** 2))
score=r2_score(y_test_opt,y_predX_opt)
print('r2 socre is:' ,score)
print('mean_sqrd_error is==',mean_squared_error(y_test_opt,y_predX_opt))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test_opt,y_predX_opt)))
 
