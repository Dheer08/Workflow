import csv
import math
import random 
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.naive_bayes import GaussianNB 
from sklearn import preprocessing,linear_model
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import sklearn
import numpy as np
#from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier
import statistics
from sklearn.ensemble import AdaBoostClassifier

data =pd.read_csv("Real Dataset.csv")
#print(data.head())

le =preprocessing.LabelEncoder()
cloudlet_ID =le.fit_transform(list(data["cloudlet ID"]))
Datacenter_ID =le.fit_transform(list(data["Data center ID"]))
VM_ID = le.fit_transform(list(data["VM ID"]))
Bwutil =le.fit_transform(list(data["Bwutil"]))
CPUutil =le.fit_transform(list(data["CPUutil"]))
memutil =le.fit_transform(list(data["memutil"]))
Disk_util =le.fit_transform(list(data["Disk util"]))
#turn_aroundTime =data["turn aroundTime"]
#Start_Time =le.fit_transform(list(data["Start Time"]))
#Finish_Time =le.fit_transform(list(data["Finish Time"]))
#namespace =le.fit_transform(list(data["namespace"]))
status =le.fit_transform(list(data["STATUS"]))

x=list(zip(cloudlet_ID,Datacenter_ID,VM_ID,Bwutil,CPUutil,memutil,Disk_util))
y=list(status)

x_train,x_test,y_train,y_test =sklearn.model_selection.train_test_split(x,y,test_size = 0.1)

model1 =RandomForestClassifier(n_estimators=10)
model2 =KNeighborsClassifier(n_neighbors=5)
model3=svm.SVC(gamma='auto')
model4=linear_model.LinearRegression()
model5=linear_model.LogisticRegression()
model6=GaussianNB()
model7=DecisionTreeClassifier()



model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)
model5.fit(x_train,y_train)
model6.fit(x_train,y_train)
model7.fit(x_train,y_train)

acc1 =model1.score(x_test,y_test)
acc2=model2.score(x_test,y_test)
acc3=model3.score(x_test,y_test)
acc4=model4.score(x_test,y_test)
acc5=model5.score(x_test,y_test)
acc6=model6.score(x_test,y_test)
acc7=model7.score(x_test,y_test)

with open('model1_pickle','wb') as pic1:
	pickle.dump(model1,pic1)

with open('model6_pickle','wb') as pic6:
	pickle.dump(model6,pic6)

with open('model7_pickle','wb') as pic7:
	pickle.dump(model7,pic7)

with open('model3_pickle','wb') as pic3:
	pickle.dump(model3,pic3)

pred1=model1.predict(x_test)
pred2=model2.predict(x_test)
pred3=model3.predict(x_test)
pred6=model6.predict(x_test)
pred7=model7.predict(x_test)


final_pred1 =np.array([])
for i in range(0,len(x_test)):
	final_pred1 =np.append(final_pred1,statistics.mean([pred1[i],pred2[i],pred3[i]]))

final_pred2 =np.array([])
for i in range(0,len(x_test)):
	final_pred2 =np.append(final_pred2,statistics.mean([pred1[i],pred2[i],pred6[i]]))

final_pred3 =np.array([])
for i in range(0,len(x_test)):
	final_pred3 =np.append(final_pred3,statistics.mean([pred1[i],pred6[i],pred3[i]]))

final_pred6 =np.array([])
for i in range(0,len(x_test)):
	final_pred6 =np.append(final_pred6,statistics.mean([pred6[i],pred2[i],pred3[i]]))

final_pred7 =np.array([])
for i in range(0,len(x_test)):
	final_pred7 =np.append(final_pred7,statistics.mean([pred6[i],pred7[i],pred1[i]]))

final_pred8 =np.array([])
for i in range(0,len(x_test)):
	final_pred8 =np.append(final_pred8,statistics.mean([pred6[i],pred7[i],pred2[i]]))

final_pred9=np.array([])
for i in range(0,len(x_test)):
	final_pred9=np.append(final_pred9,statistics.mean([pred3[i],pred6[i],pred7[i]]))

acc11=accuracy_score(y_test,final_pred1,normalize=True)
acc12=accuracy_score(y_test,final_pred2)
acc13=accuracy_score(y_test,final_pred3)
acc14=accuracy_score(y_test,final_pred6)
acc15=accuracy_score(y_test,final_pred7)
acc16=accuracy_score(y_test,final_pred8)
acc17=accuracy_score(y_test,final_pred9)

#Pre6=metrics.precision_score(y_test,pred6,average=None)

model=AdaBoostClassifier(random_state=1)
model.fit(x_train,y_train)
ac=model.score(x_test,y_test)

model12 = BaggingClassifier(DecisionTreeClassifier(random_state=1))
model12.fit(x_train,y_train)
ac12=model12.score(x_test,y_test)

model13=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
#model13.fit(x_train,y_train)
#acc13=model13.score(x_test,y_test)


print("\n\n")
print("Random Forest :",end="")
print(acc1)
print("Kneighbors :",end="")
print(acc2)
print("SVM :",end="")
print(acc3)
print("Linear Regression :",end="")
print(acc4)
print("Logistic Regression :",end="")
print(acc5)
print("Naive Bayes :",end="")
print(acc6)
print("Decision Tree :",end="")
print(acc7)
print("\nAdaBoostClassifier :",end="")
print(ac)
print("Bagging Classifier :",end="")
print(acc12)
#print("XGB Classifier :",end="")
#print(acc13)
print(" BY AVERAGE :")
print("RandomForest,KNeighbors,SVM :",end="")
print(acc11)
print("RandomForest,KNeighbors,NaiveBayes :",end="")
print(acc12)
print("RandomForest,SVM,NaiveBayes :",end="")
print(acc13)
print("SVM,KNeighbors,NaiveBayes :",end="")
print(acc14)
#print(" Precison Score Naive_bayes :",end="")
#print(Pre6)
print("NaiveBayes,DescisonTree,RandomForest :",end="")
print(acc15)
print("NaiveBayes,DecisionTree,KNeighbors :",end="")
print(acc16)
print("NaiveBayes,DecisionTree,SVM :",end="")
print(acc17)












 
