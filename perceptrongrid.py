import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import sys
nj=1
print("n_jobs="+str(nj))
df=pd.read_csv(sys.argv[1],",", header=None)
#df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
df=df.drop([0], axis=1)
df.columns=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df=df.drop("14a", axis=1)
df=df.drop("11b", axis=1)
y=df[df.columns[len(df.columns)-1]]
df=df.drop(df.columns[len(df.columns)-1], axis=1)
scaler = StandardScaler()
Xs=scaler.fit_transform(df)
cv = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=42)
model=Perceptron(tol=1e-6, random_state=0)
parameters = {"max_iter":[1,2,3,4,5,6,7,8,9,10,15,20,30,40,60,80,100,150,200],
              "penalty":["l1","l2"],
              "eta0":[0.01,0.05,0.1,0.2,0.4,0.6,0.8,1],
              "fit_intercept":[True, False],
              "alpha":[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,0.9]
              }

clf = GridSearchCV(model, parameters, cv=cv,return_train_score=True,scoring="accuracy",n_jobs=nj)
clf.fit(Xs, y)
data=pd.DataFrame(clf.cv_results_)
data=data.sort_values(by=["mean_test_score"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data)
data.to_csv("perceptronregularizedcv.csv")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  accuracy_score
import numpy as np
skf = StratifiedKFold(n_splits=4,shuffle=True)
skf.get_n_splits(Xs, y)
results=[]
for i in [1,2,3,4,5,6,7,8,9,10,15,20,30,40,60,80,100,150,200]:
    for e in [0.01,0.05,0.1,0.2,0.4,0.6,0.8,1]:
        for b in [True, False]:
            r=[i,e,b]
            test=[]
            train=[]
            for train_index, test_index in skf.split(Xs, y):
                X_train, X_test = Xs[train_index], Xs[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf = Perceptron(max_iter=i,tol=1e-8, random_state=0,penalty=None,eta0=e,fit_intercept=b)
                clf.fit(X_train, y_train)
                y_predtrain=clf.predict(X_train) 
                y_pred=clf.predict(X_test)           
                test.append(100*accuracy_score(y_test, y_pred))
                train.append(100*accuracy_score(y_train, y_predtrain))
            r.append(np.array(test).mean())
            r.append(np.array(train).mean())
            r.append(np.array(test).std())
            r.append(np.array(train).std())
            for j in range(4):
                r.append(test[j])
            for j in range(4):
                r.append(train[j])
            results.append(np.array(r).copy())
            r.clear()
            test.clear()
            train.clear()
               
results=pd.DataFrame(np.array(results))
results.columns=["iter","lr","b","mean_test_score","mean_train_score","std_test_score",
                 "std_train_score","test0","test1","test2","test3","train0","train1","train2","train3"]
results=results.sort_values(by=["mean_test_score"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(results)
results.to_csv("perceptrononecv.csv")
