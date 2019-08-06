import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import sys
import time
nj=3
print("n_jobs="+str(nj))
df=pd.read_csv(sys.argv[1],",", header=None)
#==============================================================================
#df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
#==============================================================================
df=df.drop([0], axis=1)
feat=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df.columns=feat
df=df.drop("14a", axis=1)
df=df.drop("11b", axis=1)
y=df[df.columns[len(df.columns)-1]]
X=df.drop(df.columns[len(df.columns)-1], axis=1)
scaler = StandardScaler()
Xs=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1)
results=[]
c="gini"
#for c in ["gini"]:
for n in [900,1000,1200,1400,1600,1800,2000]:
     for md in [40,50,60,70,80,90,None]:
         for ft in [0.3,0.6]:
            for mxs in [0.3,0.6,0.8]:
                for fb in [0.3,0.6,0.9]:
                    t=DecisionTreeClassifier(max_depth=md,criterion=c,max_features=ft)
                    clf=BaggingClassifier(base_estimator=t,n_estimators=n, n_jobs=nj,max_samples=mxs,max_features=fb) 
                    startTime = time.time() 
                    clf.fit(X_train,y_train)
                    y_pred=clf.predict(X_test)
                    y_predtrain=clf.predict(X_train) 
                    scoretest=100*accuracy_score(y_test, y_pred)
                    scoretrain=100*accuracy_score(y_train, y_predtrain)
                    endTime = time.time()
                    print([c,md,t.max_features,clf.max_features,mxs,n,scoretest,endTime - startTime,scoretrain])
                    results.append([c,md,t.max_features,clf.max_features,mxs,n,scoretest,endTime - startTime,scoretrain])
                
results=pd.DataFrame(np.array(results))
results.columns=["criterion","depth","tree_features","bagg_features","samples","n_estimators","test","Time","train"]
results=results.sort_values(by=["test"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(results)
results.to_csv("baggsearch.csv")
