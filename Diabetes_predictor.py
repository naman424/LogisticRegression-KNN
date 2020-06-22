#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset
X_train=pd.read_csv('Diabetes_XTrain.csv')
X_test=pd.read_csv('Diabetes_XTest.csv')
y_train=pd.read_csv('Diabetes_yTrain.csv')

#visualising the data
list=np.arange(len(y_train))
plt.scatter(list,y_train,color='red')
plt.title('Dataset')
plt.show()

#feature scaling
from sklearn.preprocessing import  StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

#logistic regression
from sklearn.linear_model import  LogisticRegression
log_classifier=LogisticRegression(random_state=0)
log_classifier.fit(X_train,y_train)

#K-NN
from sklearn.neighbors import KNeighborsClassifier
knn_classifier=KNeighborsClassifier()
knn_classifier.fit(X_train,y_train)

#predicting results
list2=np.arange(len(X_test))
log_pred=log_classifier.predict(X_test)
knn_pred=knn_classifier.predict(X_test)

#visualising the predicted results
plt.bar(list2,log_pred,color='blue')
plt.bar(list2,knn_pred,color='red')
plt.title('Diabetic or Non-Diabetic')
plt.legend( ['Logistic Regression', 'K-NN'])
plt.show()
