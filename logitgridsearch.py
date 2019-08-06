import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sys


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
cv = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=42)
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



model=LogisticRegression( C=1000000000,  penalty="l2", solver="sag",tol=0.000001)
parameters = {"max_iter":[800,900,1000,1100,1200,1300,1400,1500],
              "fit_intercept":[True, False]
              }
clf = GridSearchCV(model, parameters, cv=cv,return_train_score=True,scoring="accuracy",n_jobs=2)
clf.fit(Xs, y)
data=pd.DataFrame(clf.cv_results_)
data=data.sort_values(by=["mean_test_score"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data)
data.to_csv("classiclogit.csv")

print(" ########################################################################")   
print(" ########################################################################")   
model=LogisticRegression( penalty="l2", solver="sag",tol=0.000001)
parameters = {"max_iter":[800,900,1000,1100,1200,1300,1400,1500],
              "fit_intercept":[True, False], "C":[1,5,10,20,40,60,80,100]
              }
clf = GridSearchCV(model, parameters, cv=cv,return_train_score=True,scoring="accuracy",n_jobs=2)
clf.fit(Xs, y)
data2=pd.DataFrame(clf.cv_results_)
data2=data2.sort_values(by=["mean_test_score"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data2)
data2.to_csv("L2logit.csv")

print(" ########################################################################")   
print(" ########################################################################")

model=LogisticRegression( penalty="l1", solver="saga",tol=0.000001)
parameters = {"max_iter":[800,900,1000,1100,1200,1300,1400,1500],
              "fit_intercept":[True, False], "C":[1,5,10,20,40,60,80,100]
              }
clf = GridSearchCV(model, parameters, cv=cv,return_train_score=True,scoring="accuracy",n_jobs=2)
clf.fit(Xs, y)
data1=pd.DataFrame(clf.cv_results_)
data1=data1.sort_values(by=["mean_test_score"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data1)
data1.to_csv("L1logit.csv")
print(" ########################################################################")   
print(" ########################################################################")

X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1)           
from statsmodels.api import Logit, add_constant
print("Statistics with constant")
logit_modelb=Logit(y_train,add_constant(X_train))
result=logit_modelb.fit()
print("Summary")
print(result.summary2())
print(" ########################################################################")   
print(" ########################################################################")                   
print("Statistics without intercept")
logit_model=Logit(y_train,X_train)
result=logit_model.fit()
print("Summary")
print(result.summary2())