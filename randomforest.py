import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import  accuracy_score
nj=3 
df=pd.read_csv('/home/nestor/uk/21v/21v83k.txt',",", header=None)
#df=pd.read_csv(sys.argv[1],",", header=None)
df=df.drop([0], axis=1)
feat=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df.columns=feat
df=df.drop("14a", axis=1)
df=df.drop("11b", axis=1)
y=df[df.columns[len(df.columns)-1]]
X=df.drop(df.columns[len(df.columns)-1], axis=1)


scaler = StandardScaler()
Xs=scaler.fit_transform(X)
##shuffle is for randomize the data , default true.
Xs=pd.DataFrame(Xs)
Xs.columns=X.columns
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1)
scaler = StandardScaler()
Xs=scaler.fit_transform(X)
##shuffle is for randomize the data , default true.

X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100, n_jobs=nj)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
print(clf.get_params)
# prediction on test set
y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation

# Model Accuracy, how often is the classifier correct?
print("Accuracy RF:", accuracy_score(y_test, y_pred))


#Create a Gaussian Classifier
clf=ExtraTreesClassifier(n_estimators=100, n_jobs=nj)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
print(clf.get_params)
# prediction on test set
y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation

# Model Accuracy, how often is the classifier correct?
print("AccuracyExtra :", accuracy_score(y_test, y_pred))
