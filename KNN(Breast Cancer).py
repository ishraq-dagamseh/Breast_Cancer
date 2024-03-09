# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 18:19:33 2021

@author: Ishraq
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


#encoding the dependent variables(Because y in string type)
labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y)



#splitting X&y into training & testing
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

#feature scaling( col 1) from x
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
X_train=SC.fit_transform(X_train)
X_test=SC.fit_transform(X_test)

# Training the K-NN model on the Training set


from sklearn.neighbors import KNeighborsClassifier

Classifier= KNeighborsClassifier(n_neighbors= 5, metric='minkowski', p=2)


Classifier.fit(X_train,y_train)

y_pred=Classifier.predict(X_test)

# evalute using CM

from sklearn.metrics import confusion_matrix,classification_report


CM_KNN = confusion_matrix(y_test,y_pred)
print(CM_KNN)

print(classification_report(y_test, y_pred, target_names=['M','B']))

import seaborn as sns
sns.heatmap(CM_KNN, annot=True)

# visualizae 

# # Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                       np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, Classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#               alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('K-NN (Test set)')
# plt.xlabel('Age')
# plt.ylabel('diagnosis')
# plt.legend()
# plt.show()
