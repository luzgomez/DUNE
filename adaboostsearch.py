import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
import sys
nj=2
print("n_jobs="+str(nj))
df=pd.read_csv(sys.argv[1],",", header=None)
#==============================================================================
#df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
#==============================================================================
df=df.drop([0], axis=1)
df.columns=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df=df.drop(["11b","14a"], axis=1)
y=df[df.columns[len(df.columns)-1]]
X=df.drop(df.columns[len(df.columns)-1], axis=1)
scaler = StandardScaler()
Xs=scaler.fit_transform(X)
param_grid = {    
 'n_estimators': [300,400,500,600,700,800,900],
 'learning_rate' : [0.2,0.3,0.4,0.5,0.6,0.7],
 'base_estimator__max_depth': [1,2,3,4,5,6,7],    
 'base_estimator__criterion': ["gini","entropy"],
 'base_estimator__max_features' : [1,2,3,4,5,7,10,13]
}

algorithmValue='SAMME.R'
cv = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=42)
model=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),algorithm=algorithmValue)
clf=GridSearchCV(model,param_grid=param_grid, cv=cv,n_jobs=nj)
clf.fit(Xs, y)
data=pd.DataFrame(clf.cv_results_)
data=data.sort_values(by=["mean_test_score"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data)
data.to_csv("adaboosearch.csv")

