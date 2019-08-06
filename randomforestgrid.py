import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import sys
nj=9
print("n_jobs="+str(nj))
df=pd.read_csv(sys.argv[1],",", header=None)
#==============================================================================
# df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
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
Xs=pd.DataFrame(Xs)
Xs.columns=X.columns
scaler = StandardScaler()
Xs=scaler.fit_transform(X)
#bootstrap : boolean, optional (default=True)
model=RandomForestClassifier()
parameters = {"criterion":["gini","entropy"],
              "max_depth":[40,45,50,55,60,65,70,75,80],
              "max_features":[2,3,4,5,6],
              "n_estimators":[250,300,400,500,600,700,800,900,1000]}

clf = GridSearchCV(model, parameters, cv=4,return_train_score=True,scoring="accuracy",n_jobs=nj)
clf.fit(Xs, y)
data=pd.DataFrame(clf.cv_results_)
data=data.sort_values(by=["mean_test_score"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data)
data.to_csv("rfsearchfeat.csv")