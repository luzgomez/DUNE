import pandas as pd
#import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time
import sys
nj=4
print("n_jobs="+str(nj))
df=pd.read_csv(sys.argv[1],",", header=None)
#df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
df=df.drop([0], axis=1)
df.columns=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df=df.drop(["11b","14a"], axis=1)
y=df[df.columns[len(df.columns)-1]]
df=df.drop(df.columns[len(df.columns)-1], axis=1)
scaler = StandardScaler()
Xs=scaler.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1)
for d in ["uniform", "distance"]:
    for p in [1,2,3,4,5,6]:
        for i in [1,2,3,4,5,10,20,50,70,90,100,150,200,250,300,350,400,450,500,600,700,800,900,1000]:
            kn= KNeighborsClassifier(n_neighbors=i, weights=d, p=p, algorithm='kd_tree',n_jobs=nj)
            print("###################################")
            startTime = time.time()
            kn.fit(X_train, y_train)
            y_pred=kn.predict(X_test)           
            print(d+" n="+str(i)+"p="+str(p))
            print("Accurancy knn ="+ str(100*accuracy_score(y_test, y_pred)))
            endTime = time.time()
            print("fit time")
            print(endTime - startTime)