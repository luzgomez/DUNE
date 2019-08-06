import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import tree
import sys
nj=1
#df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
df=pd.read_csv(sys.argv[1],",", header=None)
df=df.drop([0], axis=1)
df.columns=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df=df.drop(["11b","14a"], axis=1)
y=df[df.columns[len(df.columns)-1]]
X=df.drop(df.columns[len(df.columns)-1], axis=1)
scaler = StandardScaler()
Xs=scaler.fit_transform(X)
cv = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=42)
model= tree.DecisionTreeClassifier()
parameters = {"criterion":["gini","entropy"], "max_depth":[30,40,45,50,55,60,65,70,75,80,85,90,95,100,110],"max_features":[2,4,7,10,12,14,16,18] }
cv = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=42)
clf = GridSearchCV(model, parameters, cv=cv,return_train_score=True,scoring="accuracy",n_jobs=nj)
clf.fit(Xs, y)
data=pd.DataFrame(clf.cv_results_)
data=data.drop(['params'],axis=1)
data=data.sort_values(by=["mean_test_score"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data)
data.to_csv("treesearch.csv")
