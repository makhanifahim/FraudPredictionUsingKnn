import imp
import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

data=pd.read_csv('Bank_Transaction.csv')
test=pd.read_csv('test.csv')
print(data.info())
print(data.describe())
#there are some null in data to fill it as 0 we use fillna
data=data.fillna(0)
print(data.isnull().sum())

print(data.head())

#to check th distribution of data set for froud (its 0.108%)
print(data.groupby(['isFraud']).count()/data.shape[0])

X=np.array(data.drop(['isFraud','nameOrig','nameDest'],axis=1))
y=np.array(data['isFraud'])
Xt=np.array(test.drop(['isFraud','nameOrig','nameDest'],axis=1))
yt=np.array(data['isFraud'])


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.10)
#declaration of model
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)
predict = knn.predict(X_test)

print("Accurracy",accuracy_score(y_test,predict))
y_pred= knn.predict(X_test)

Ptrue=0
Ftest=0
Total_test=0
total_real_true=0
for i in range(len(y_test)):
    if((y_test[i]==1.0 and y_pred[i]==0.0)or(y_test[i]==0.0 and y_pred[i]==1.0)):
        Ftest=Ftest+1
    Total_test=Total_test+1
    if(y_test[i]==1.0 and y_pred[i]==1.0):
        Ptrue=Ptrue+1
    if(y_test[i]==1.0):
        total_real_true=total_real_true+1

print("Total test Data = ",Total_test)
print("Total true test Data = ",Ptrue)
print("Total False Result Data = ",Ftest)
print("Total real True",total_real_true)

yPred=knn.predict(Xt)
print(yPred)