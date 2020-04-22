import pickle
from sklearn.naive_bayes import GaussianNB 
from sklearn import preprocessing,linear_model
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import sklearn
import pickle
import numpy as np
import pandas as pd
import statistics
import xlrd
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)

data=pd.read_csv("Real Dataset.csv")

le =preprocessing.LabelEncoder()
cloudlet_ID =le.fit_transform(list(data["cloudlet ID"]))
Datacenter_ID =le.fit_transform(list(data["Data center ID"]))
VM_ID = le.fit_transform(list(data["VM ID"]))
Bwutil =le.fit_transform(list(data["Bwutil"]))
CPUutil =le.fit_transform(list(data["CPUutil"]))
memutil =le.fit_transform(list(data["memutil"]))
Disk_util =le.fit_transform(list(data["Disk util"]))
#turn=le.fit_transform(list(data["turn"]))
#Start_Time =le.fit_transform(list(data["Start Time"]))
#Finish_Time =le.fit_transform(list(data["Finish Time"]))
#namespace =le.fit_transform(list(data["namespace"]))
status =le.fit_transform(list(data["STATUS"]))

x=list(zip(cloudlet_ID,Datacenter_ID,VM_ID,Bwutil,CPUutil,memutil,Disk_util))
y=list(status)

x_train,x_test,y_train,y_test =sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
#x_test=data.iloc[1:300,:7]

f=open('model1_pickle','rb')
model1=pickle.load(f)
f.close()

f=open('model6_pickle','rb')
model6=pickle.load(f)
f.close()

f=open('model7_pickle','rb')
model7=pickle.load(f)
f.close()

f=open('model3_pickle','rb')
model3=pickle.load(f)
f.close()


pred1=model1.predict(x_test)
#acc=accuracy_score(y_test,pred1)
#print("Random Forest :",end="")
#print(acc)

pred6=model6.predict(x_test)
#acc=accuracy_score(y_test,pred6)
#print("Naive bayes :",end="")
#print(acc)

pred7=model7.predict(x_test)
#acc=accuracy_score(y_test,pred7)
#print("Descision Tree :",end="")
#print(acc)

pred3=model3.predict(x_test)


final_pred1 =np.array([])
for i in range(0,len(x_test)):
	final_pred1 =np.append(final_pred1,statistics.mean([pred1[i],pred6[i],pred7[i]]))


final_pred2 =np.array([])
for i in range(0,len(x_test)):
	final_pred2 =np.append(final_pred2,statistics.mean([pred1[i],pred6[i],pred3[i]]))


final_pred3 =np.array([])
for i in range(0,len(x_test)):
	final_pred3 =np.append(final_pred3,statistics.mean([pred3[i],pred6[i],pred7[i]]))


final_ans =np.array([])
ans=0.0;
for i in range(0,len(x_test)):
	if(statistics.mean([final_pred3[i]+0.0,final_pred2[i]+0.0,final_pred1[i]+0.0,pred6[i]+0.0,pred7[i]+0.0,pred1[i]+0.0])>0.5):
		final_ans=np.append(final_ans,1)
	else:
		final_ans=np.append(final_ans,0)



#df2=pd.read_csv('Result.csv',header=None)

df=list(zip(final_pred3,final_pred2,final_pred1,pred6,pred7,pred1,final_ans))

dataframe = pd.DataFrame(df)
dataframe.columns=["A","B","C","D","E","F","G"]
dataframe.to_csv('Result.csv')





	

#acc11=accuracy_score(final_pred1,y_test)
#print("Random Forest,Naive Bayes,Descision Tree :",end="")

#print(acc11)

#acc12=accuracy_score(final_pred1,y_test)
#print("Random Forest,Naive Bayes,SVM :",end="")
#print(acc12)

#acc13=accuracy_score(final_pred1,y_test)
#print("SVM,Naive Bayes,Descision Tree :",end="")
#print(acc13)


#print(final_ans)


