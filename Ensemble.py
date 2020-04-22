import csv
import math
import random 
import pandas as pd
from sklearn.naive_bayes import GaussianNB ,BernoulliNB
from sklearn import preprocessing,linear_model
import sklearn
import numpy as np
#from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier
import statistics

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
turn_aroundTime =data["turnAround"]
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
model8=BernoulliNB()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)
model5.fit(x_train,y_train)
model6.fit(x_train,y_train)
model7.fit(x_train,y_train)
model8.fit(x_train,y_train)

acc1 =model1.score(x_test,y_test)
acc2=model2.score(x_test,y_test)
acc3=model3.score(x_test,y_test)
acc4=model4.score(x_test,y_test)
acc5=model5.score(x_test,y_test)
acc6=model6.score(x_test,y_test)
acc7=model7.score(x_test,y_test)
acc8=model8.score(x_test,y_test)

#final_pred =np.array([])
#for i in range(0,len(x_test)):
	#final_pred =np.append(final_pred,statistics.mode([pred1[i],pred2[i],pred3[i]]))

model11 =VotingClassifier(estimators=[('rf',model1),('kn',model2),('svm',model3)],voting ='hard')
#model12=VotingClassifier(estimators=[('rf',model1),('kn',model2),('lr',model4)],voting='hard')
model13=VotingClassifier(estimators=[('rf',model1),('kn',model2),('lr',model5)],voting='hard')
model14=VotingClassifier(estimators=[('rf',model1),('kn',model2),('nb',model6)],voting='hard')
model15=VotingClassifier(estimators=[('rf',model1),('svm',model3),('nb',model6)],voting='hard')
model16=VotingClassifier(estimators=[('rf',model1),('lr',model5),('nb',model6)],voting='hard')
model17=VotingClassifier(estimators=[('svm',model3),('kn',model2),('nb',model6)],voting='hard')
model18=VotingClassifier(estimators=[('lr',model5),('kn',model2),('svm',model3)],voting='hard')
#Left model12 due to conversion error and the accuracy is very low

model11.fit(x_train,y_train)
#model12.fit(x_train,y_train)
model13.fit(x_train,y_train)
model14.fit(x_train,y_train)
model15.fit(x_train,y_train)
model16.fit(x_train,y_train)
model17.fit(x_train,y_train)
model18.fit(x_train,y_train)

acc11=model11.score(x_test,y_test)
#acc12=model12.score(x_test,y_test)
acc13=model13.score(x_test,y_test)
acc14=model14.score(x_test,y_test)
acc15=model15.score(x_test,y_test)
acc16=model16.score(x_test,y_test)
acc17=model17.score(x_test,y_test)
acc18=model18.score(x_test,y_test)

print("\n\n\n")
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
print("Bernoulli NaiveBayes :",end="")
print(acc8)
print("Decision Tree :",end="")
print(acc7)
print("\n BY USING MAX VOTING")
print("RandomForest,KNeighbors,SVM :",end ="")
print(acc11)
#print("RandomForest,KNeighbors,LinearRegression :",end="")
#print(acc12)
print("RandomForest,KNeighbors,LogisticRegression :",end="")
print(acc13)
print("RandomForest,KNeighbors,NaiveBayes :",end="")
print(acc14)
print("RandomForest,SVM,Naive Bayes :",end="")
print(acc15)
print("RandomForest,LogisticRegression,Naive Bayes :",end="")
print(acc16)
print("SVM,KNeighbors,NaiveBayes :",end="")
print(acc17)
print("LogisticRegression,KNeighbors,SVM :",end="")
print(acc18)










 
