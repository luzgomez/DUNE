import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
import sys
import time
nj=3
print("n_jobs="+str(nj))
df=pd.read_csv(sys.argv[1],",", header=None)
#df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)

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

print("######################################")

p="l1"
a=0.0001
b=True
e=0.01
ite=1
model=Perceptron(max_iter=ite,tol=1e-8, penalty="l1",eta0=e,alpha=a,fit_intercept=b)
print(model.get_params)
paramsfull=0
scorefull=0
print("Full feature")
for i in [50,80, 100,200,300,400,500,600,700,800,900,1000]:
    bagg = BaggingClassifier(model, max_samples=1.0, max_features=1.0, n_estimators=i, n_jobs=nj)
    
    print("######################################")
    print(bagg.get_params)
    print("ntrees_bagging:",i)
    startTime = time.time() 
    bagg.fit(X_train, y_train)
    ypred=bagg.predict(X_test)
    score=100*accuracy_score(y_test, ypred)
    print("accurancy test="+str(score))
    ypredtrain=bagg.predict(X_train)
    print("accurancy train="+str(100*accuracy_score(y_train, ypredtrain)))
    endTime=time.time()
    print("fit time="+str(endTime - startTime))
    if score>scorefull:
        scorefull=score
        paramsfull=i

print("######################################")
print("######################################")
print("######################################")
params4=0
score4=0
print("4 feature")
for i in [50,80, 100,200,300,400,500,600,700,800,900,1000]:
    bagg = BaggingClassifier(model, max_samples=1.0, max_features=1/(4.5), n_estimators=i, n_jobs=nj)
    
    print("######################################")
    print(bagg.get_params)
    print("ntrees_bagging:",i)
    startTime = time.time() 
    bagg.fit(X_train, y_train)
    ypred=bagg.predict(X_test)
    score=100*accuracy_score(y_test, ypred)
    print("accurancy test="+str(score))
    ypredtrain=bagg.predict(X_train)
    print("accurancy train="+str(100*accuracy_score(y_train, ypredtrain)))
    endTime=time.time()
    print("fit time="+str(endTime - startTime))
    if score>score4:
        score4=score
        params4=i

print("######################################")
print("######################################")
print("######################################")

params41=0
score41=0
print("4 feature and 60% of the data")
for i in [50,80, 100,200,300,400,500,600,700,800,900,1000]:
    bagg = BaggingClassifier(model, max_samples=0.6, max_features=1/(4.5), n_estimators=i, n_jobs=nj)
    
    print("######################################")
    print(bagg.get_params)
    print("ntrees_bagging:",i)
    startTime = time.time() 
    bagg.fit(X_train, y_train)
    ypred=bagg.predict(X_test)
    score=100*accuracy_score(y_test, ypred)
    print("accurancy test="+str(score))
    ypredtrain=bagg.predict(X_train)
    print("accurancy train="+str(100*accuracy_score(y_train, ypredtrain)))
    endTime=time.time()
    print("fit time="+str(endTime - startTime))
    if score>score41:
        score41=score
        params41=i

print("######################################")
print("######################################")
print("######################################")


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("n_jobs="+str(nj))
print("###################################")
print("paramsfull")
print(paramsfull)
print("maxscorefull")
print(scorefull)
print("###################################")
print("params4feature")
print(params4)
print("maxscore4feature")
print(score4)
print("###################################")
print("params4feature0.6data")
print(params41)
print("maxscore4feature0.6data")
print(score41)