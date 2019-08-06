import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
import time

nj=4
#df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
df=pd.read_csv(sys.argv[1],",", header=None) 
########################################################################
df=df.drop([0], axis=1)
df.columns=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df=df.drop("14a", axis=1)
df=df.drop("11b", axis=1)
y=df[df.columns[len(df.columns)-1]]
df=df.drop(df.columns[len(df.columns)-1], axis=1)
scaler = StandardScaler()
Xs=scaler.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1)
########################################################################

#fit_intercept : bool, default: True Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

#For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
#For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
#newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas ‘liblinear’ and ‘saga’ handle L1 penalty.
#Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. 

#C=1 default 1, smaller there is a penalty for increasing the magnitude of the parameters. Conversely, there 
#tends to be a benefit to decreasing the magnitude of the parameters. Avoid overfiting in case of small data size
# same idea of svm

#‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, 
#whereas ‘liblinear’ and ‘saga’ handle L1 penalty.

#The key difference between these techniques is that Lasso l1 shrinks the less important feature’s coefficient to zero thus,
# removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.

clf=LogisticRegression()
paramsnone=clf.get_params
scorenone=0
p="l2"
for b in [True, False]:
    for i in [5,10,20,40,60,80,100,200,300,400,500,600,700,800,900,1000, 1200,1400,1600,1800,2000]:
        print("###################################")
        clf = LogisticRegression( C=1000000000,fit_intercept=b,max_iter=i,  penalty=p, solver="sag",tol=0.00001,n_jobs=nj)
        startTime = time.time()
        clf.fit(X_train, y_train)
        y_predtrain=clf.predict(X_train) 
        y_pred=clf.predict(X_test)           
        score1=100*accuracy_score(y_test, y_pred)
        print("Accurancy  test Logistic ="+ str(score1))
        print("Accurancy  train perceptron ="+ str(100*accuracy_score(y_train, y_predtrain)))
        print("max iter="+str(i)+" penalty="+p+" intercept="+str(b))
        endTime = time.time()
        if score1>scorenone:
            scorenone=score1
            paramsnone=clf.get_params
        print("fit time="+str(endTime - startTime))
print("##############################################################################")
print("##############################################################################")
print("##############################################################################")
print("##############################################################################")

clf=LogisticRegression()
l2params=clf.get_params
l2score=0
penalty=['l2','l1']
solver=['sag','saga']
p="l2"
solv="sag"
for b in [True, False]:
    for i in [5,10,40,60,80,100,200,300,400,500,600,700,800,900,1000, 1200,1400,1600,1800,2000]:
        for c in [0.1,0.5,1,5,10,20,40,60,80,100]:
            print("###################################")
            clf = LogisticRegression( C=c,fit_intercept=b,max_iter=i,  penalty=p, solver=solv,tol=0.00001,n_jobs=nj)
            startTime = time.time()
            clf.fit(X_train, y_train)
            y_predtrain=clf.predict(X_train) 
            y_pred=clf.predict(X_test)           
            score1=100*accuracy_score(y_test, y_pred)
            print("Accurancy  test Logistic ="+ str(score1))    
            print("Accurancy  train Logistic ="+ str(100*accuracy_score(y_train, y_predtrain)))
            print("max iter="+str(i)+" penalty="+p+" intercept="+str(b)+" C="+str(c))
            endTime = time.time()
            if score1>l2score:
                l2score=score1
                l2params=clf.get_params
            print("fit time="+str(endTime - startTime))
print("##############################################################################")
print("##############################################################################")
print("##############################################################################")
print("##############################################################################")

clf=LogisticRegression()
l1params=clf.get_params
l1score=0
p="l1"
solv="saga"
for b in [True, False]:
     for i in [5,10,20,40,60,80,100,200,300,400,500,600,700,800,900,1000, 1200,1400,1600,1800,2000]:
        for c in [0.1,0.5,1,5,10,20,40,60,80,100]:
            print("###################################")
            clf = LogisticRegression( C=c,fit_intercept=b,max_iter=i,  penalty=p, solver=solv,tol=0.00001,n_jobs=nj)
            startTime = time.time()
            clf.fit(X_train, y_train)
            y_predtrain=clf.predict(X_train) 
            y_pred=clf.predict(X_test)           
            score1=100*accuracy_score(y_test, y_pred)
            print("Accurancy  test Logistic ="+ str(score1))    
            print("Accurancy  train perceptron ="+ str(100*accuracy_score(y_train, y_predtrain)))
            print("max iter="+str(i)," penalty="+p+" intercept="+str(b)+" C="+str(c))
            endTime = time.time()
            if score1>l1score:
                l1score=score1
                l1params=clf.get_params
            print("fit time="+str(endTime - startTime))
print("##############################################################################")
print("##############################################################################")
print("##############################################################################")
print("##############################################################################")
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


           
from statsmodels.api import Logit, add_constant
print("Statistics with constant")
logit_modelb=Logit(y_train,add_constant(X_train))
result=logit_modelb.fit()
print("Summary")
print(result.summary2())

print(" ########################################################################")                   
print("Statistics without intercept")
logit_model=Logit(y_train,X_train)
result=logit_model.fit()
print("Summary")
print(result.summary2())











