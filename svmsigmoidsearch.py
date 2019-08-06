import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.svm import SVC
import sys
import time
nj=3
print("n_jobs="+str(nj))
df=pd.read_csv(sys.argv[1],",", header=None)
#==============================================================================
#df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
#https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf (We recommend a “grid-search)
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
for c in [10**i for i in range(-4,9)]:
    for g in np.logspace(-10, 4, 15):
        for b in [-16 + 4*i for i in range(9)]:             
                        clf = SVC(kernel='poly',random_state=0,C=c,gamma=g, coef0=b) 
                        startTime = time.time() 
                        clf.fit(X_train,y_train)
                        y_pred=clf.predict(X_test)
                        y_predtrain=clf.predict(X_train) 
                        scoretest=100*accuracy_score(y_test, y_pred)
                        scoretrain=100*accuracy_score(y_train, y_predtrain)
                        endTime = time.time()
                        results.append([c,g,b,scoretest,endTime - startTime,scoretrain])
                        
                
results=pd.DataFrame(np.array(results))
results.columns=["C","gamma","bias","test","Time","train"]
results=results.sort_values(by=["test"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(results)
results.to_csv("svmsigmoidsearch.csv")
