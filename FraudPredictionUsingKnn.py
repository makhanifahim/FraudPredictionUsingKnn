import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv('Bank_Transaction.csv')
test=pd.read_csv('test.csv')
print(data.info())
print(data.describe())
#there are some null in data to fill it as 0 we use fillna
data=data.fillna(0)

#to check th distribution of data set for froud (its 0.108%)
print(data.groupby(['isFraud']).count()/data.shape[0])

X=data.drop(['isFraud','nameOrig','nameDest'],axis=1)
y=data['isFraud']
Xt=test.drop(['isFraud','nameOrig','nameDest'],axis=1)
yt=test['isFraud']


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.10)
#declaration of model
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)
predict = knn.predict(X_test)

print("Accurracy",accuracy_score(y_test,predict))

print("--------------------Multiple prediction----------------------------")

yPred=knn.predict(Xt)

print("--------------------Single prediction----------------------------")

#columns it is asking for
# Dont pass nameOrig,nameDest and isFraud in it
# #   step,type,amount,oldbalanceOrg,oldbalanceDest,newbalanceDest,isFlaggedFraud
NewData1=[ 1,1,9839.64,170136.00,160296.36,0.00,0.00,0]
NewData2=[84,2,8380.79,8380.79,0,0,0,0]
predict_single=knn.predict([np.array(NewData2)])

print("prediction of given data is = ",predict_single[0])

if(predict_single[0]==0):
    print("Dont worry there is no Fraud")
else:
    print("There is Fraud")