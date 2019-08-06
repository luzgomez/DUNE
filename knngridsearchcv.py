import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
import sys
nj=2
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
cv = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=42)
model= KNeighborsClassifier(weights="distance", algorithm='kd_tree',n_jobs=nj)
parameters = {"n_neighbors":[50,60,70,80,90], "p":[1,2] }
cv = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=42)
clf = GridSearchCV(model, parameters, cv=cv,return_train_score=True,scoring="accuracy",n_jobs=nj)
clf.fit(Xs, y)
data=pd.DataFrame(clf.cv_results_)
data=data.drop(['params'],axis=1)
data=data.sort_values(by=["mean_test_score"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data)
data.to_csv("knnsearchcv.csv")