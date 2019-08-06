import pandas as pd
#import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.linear_model import Perceptron
import time
import sys
nj=4
print("n_jobs="+str(nj))
#df=pd.read_csv(sys.argv[1],",", header=None)
df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
df=df.drop([0], axis=1)
df.columns=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df=df.drop("14a", axis=1)
df=df.drop("11b", axis=1)
y=df[df.columns[len(df.columns)-1]]
df=df.drop(df.columns[len(df.columns)-1], axis=1)
scaler = StandardScaler()
Xs=scaler.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1)
#penalty : None, ‘l2’ or ‘l1’ or ‘elasticnet’
#alpha : float    Constant that multiplies the regularization term if regularization is used. Defaults to 0.0001
#max_iter : int, optional (default=1000)
#tol : float or None, optional (default=1e-3)
#shuffle : bool, optional, default True
#eta0 : double  Constant by which the updates are multiplied. Defaults to 1.
clf=Perceptron()
paramsnone=clf.get_params
scorenone=0
p="none"
for b in [True, False]:
    for e in [0.01,0.05,0.1,0.2,0.4,0.6,0.8,1]:
        for i in [1,2,3,4,5,6,10,40,60,80,500,100,200,300,400,500,600,700,800,900,1000, 1200,1400,1600,1800,2000,2200,2400,2600,3000,4000,5000,6000]:
            print("###################################")
            clf = Perceptron(max_iter=i,tol=1e-8, random_state=0,penalty=p,eta0=e,fit_intercept=b)
            startTime = time.time()
            clf.fit(X_train, y_train)
            y_predtrain=clf.predict(X_train) 
            y_pred=clf.predict(X_test)           
            score1=100*accuracy_score(y_test, y_pred)
            print("Accurancy  test perceptron ="+ str(score1))
            print("Accurancy  train perceptron ="+ str(100*accuracy_score(y_train, y_predtrain)))
            print("max iter="+str(i),"penalty="+p+"learning rate="+str(e))
            endTime = time.time()
            if score1>scorenone:
                scorenone=score1
                paramsnone=clf.get_params
            print("fit time")
            print(endTime - startTime)
print("##############################################################################")
print("##############################################################################")
print("##############################################################################")
print("##############################################################################")
#
clf=Perceptron()
l1params=clf.get_params
l1score=0
p="l1"
for b in [True, False]:
    for e in [0.01,0.05,0.1,0.2,0.4,0.6,0.8,1]:
       for i in [1,2,3,4,5,6,10,40,60,80,500,100,200,300,400,500,600,700,800,900,1000, 1200,1400,1600,1800,2000,2200,2400,2600,3000,4000,5000,6000]:         
            for a in [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,0.9]:
                print("###################################")
                clf = Perceptron(max_iter=i,tol=1e-8, random_state=0,penalty=p,eta0=e,alpha=a,fit_intercept=b)
                startTime = time.time()
                clf.fit(X_train, y_train)
                y_predtrain=clf.predict(X_train) 
                y_pred=clf.predict(X_test)           
                score1=100*accuracy_score(y_test, y_pred)
                print("Accurancy  test perceptron ="+ str(score1))
                print("Accurancy  train perceptron ="+ str(100*accuracy_score(y_train, y_predtrain)))
                print("max iter="+str(i),"penalty="+p+" learning rate="+str(e),"regularization term="+str(a))
                if score1>l1score:
                    l1score=score1
                    l1params=clf.get_params
                endTime = time.time()
                print("fit time")
                print(endTime - startTime)
print("##############################################################################")
print("##############################################################################")
print("##############################################################################")
print("##############################################################################")                    


clf=Perceptron()
l2params=clf.get_params
l2score=0
p="l2"
for b in [True, False]:
    for e in [0.01,0.05,0.1,0.2,0.4,0.6,0.8,1]:
        for i in [1,2,3,4,5,6,10,40,60,80,500,100,200,300,400,500,600,700,800,900,1000, 1200,1400,1600,1800,2000,2200,2400,2600,3000,4000,5000,6000]:        
            for a in [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,0.9]:
                print("###################################")
                clf = Perceptron(max_iter=i,tol=1e-8, random_state=0,penalty=p,eta0=e,alpha=a,fit_intercept=b)
                startTime = time.time()
                clf.fit(X_train, y_train)
                y_predtrain=clf.predict(X_train) 
                y_pred=clf.predict(X_test)           
                score1=100*accuracy_score(y_test, y_pred)
                print("Accurancy  test perceptron ="+ str(score1))
                print("Accurancy  train perceptron ="+ str(100*accuracy_score(y_train, y_predtrain)))
                print("max iter="+str(i),"penalty="+p+" learning rate="+str(e),"regularization term="+str(a))
                if score1>l2score:
                    l2score=score1
                    l2params=clf.get_params
                endTime = time.time()
                print("fit time")
                print(endTime - startTime)
                    

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("n_jobs="+str(nj))
print("###################################")
print("params")
print(paramsnone)
print("maxscore")
print(scorenone)
print("###################################")
print("params")
print(l2params)
print("maxscore")
print(l2score)
print("###################################")
print("params")
print(l1params)
print("maxscore")
print(l1score)