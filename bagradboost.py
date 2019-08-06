import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
#import datetime as time
#df=pd.read_csv(sys.argv[1],",", header=None)
df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)

#-------------------------------df=pd.read_csv('/usera/nap47/Desktop/nwdata/21v.txt',",", header=None)


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

print("######################################")
n_estimatorsValue=200
learning_rateValue=0.4
lossfunction='deviance'
max_depthValue=3
minsamplesleaf=0.05 
validfrac=0.1
niternochange=None
subsam=0.6

gbtmodel = GradientBoostingClassifier( loss=lossfunction, n_estimators=n_estimatorsValue, learning_rate=learning_rateValue, 
                                      max_depth=max_depthValue, n_iter_no_change=niternochange, validation_fraction=validfrac,subsample=subsam)

for i in [50,80, 100, 130,160,190,200,230, 260,280,300,330,360,390,400]:
    bagg = BaggingClassifier(gbtmodel, max_samples=1.0, max_features=1.0, n_estimators=i, n_jobs=4)
    
    print("######################################")
    print(bagg.get_params)
    print("ntrees_bagging:",i)
    #startTime = time.time() 
    bgada=bagg.fit(X_train, y_train)
    #endTime = time.time()
    #print("fit time")
    #print(endTime - startTime)
    ypred=bgada.predict(X_test)
    print("accurancy test")
    accbdt=accuracy_score(y_test, ypred)
    print(accbdt)
    ypredtrain=bgada.predict(X_train)
    print("accurancy train")
    accbdtrain=accuracy_score(y_train, ypredtrain)
    print(accbdtrain)