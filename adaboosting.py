import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from statsmodels.api import Logit, add_constant
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import sys
import datetime as time
#df=pd.read_csv(sys.argv[1],",", header=None)
df=pd.read_csv('/home/nestor/uk/21v/21v.txt',",", header=None)
#df=pd.read_csv('/home/nestor/uk/data1k.txt',",", header=None)

#-------------------------------df=pd.read_csv('/usera/nap47/Desktop/nwdata/21v.txt',",", header=None)

#cross=5
#gridge=28
#stepoch=10
#epoch=210
col=['g','r'] 
df=df.drop([0], axis=1)
feat=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df.columns=feat
df=df.drop("14a", axis=1)
df=df.drop("11b", axis=1)
y=df[df.columns[len(df.columns)-1]]
X=df.drop(df.columns[len(df.columns)-1], axis=1)


scaler = StandardScaler()
Xs=scaler.fit_transform(X)
##shuffle is for randomize the data , default true.
Xs=pd.DataFrame(Xs)
Xs.columns=X.columns
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1)
print("800 tunning=tmva")
print("######################################")
n_estimatorsValue=200
learning_rateValue=0.9
algorithmValue='SAMME.R'
max_depthValue=3
minsamplesleaf=0.05 

mlmodel=DecisionTreeClassifier(max_depth=max_depthValue,min_samples_leaf=minsamplesleaf )
bdtModel = AdaBoostClassifier(mlmodel, n_estimators=n_estimatorsValue, learning_rate=learning_rateValue, algorithm=algorithmValue) 
print(bdtModel.get_params)
#startTime = time.time() 
bdtfit=bdtModel.fit(X_train, y_train)
#endTime = time.time()
#print("fit time")
#print(endTime - startTime)
ypred=bdtModel.predict(X_test)
print("accurancy test")
accbdt=accuracy_score(y_test, ypred)
print(accbdt)
ypredtrain=bdtModel.predict(X_train)
print("accurancy train")
accbdtrain=accuracy_score(y_train, ypredtrain)
print(accbdtrain)


#lr=0.9




