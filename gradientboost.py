import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import sys

#df=pd.read_csv(sys.argv[1],",", header=None)

df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
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
print("100 tunning=tmva")
print("######################################")

n_estimatorsValue=200
learning_rateValue=0.4
#lossfunction='exponential'
lossfunction='deviance'
max_depthValue=3
minsamplesleaf=0.05 
validfrac=0.1
niternochange=None
subsam=0.6
verb=1
gbtmodel = GradientBoostingClassifier( loss=lossfunction, n_estimators=n_estimatorsValue, learning_rate=learning_rateValue, 
                                      max_depth=max_depthValue, n_iter_no_change=niternochange, validation_fraction=validfrac,subsample=subsam,verbose=verb) 
print(gbtmodel.get_params)
#startTime = time.time() 
gbtfit=gbtmodel.fit(X_train, y_train)
#endTime = time.time()
#print("fit time")
#print(endTime - startTime)

ypred=gbtmodel.predict(X_test)
gbtcofmat=confusion_matrix(y_test, ypred)
print("Confusion Matrix test")
print(gbtcofmat)
print("accurancy test")
accgbt=accuracy_score(y_test, ypred)
print(accgbt)
ypredtrain=gbtmodel.predict(X_train)
gbtcofmatrain=confusion_matrix(y_train, ypredtrain)
print("Confusion Matrix train")
print(gbtcofmatrain)
print("accurancy train")
accgbtrain=accuracy_score(y_train, ypredtrain)
print(accgbtrain)
varimport=pd.DataFrame(gbtmodel.feature_importances_).transpose()
varimport.columns=X.columns
varimport=varimport*100
print(varimport.sort_values(by=0, ascending=False, axis=1).transpose())

#0.5lr







