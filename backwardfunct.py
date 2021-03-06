import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mlt
import sys
import time

#df=pd.read_csv(sys.argv[1],",", header=None)
df=pd.read_csv('/home/nestor/uk/21v/21v83k.txt',",", header=None)
df=df.drop([0], axis=1)
df.columns=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df=df.drop("14a", axis=1)
df=df.drop("11b", axis=1)
def lineplot(history_feature_score, string):
    plt.rcParams.update({'font.size': 6})
    plt.rcParams['lines.linewidth'] = 0.8
    mlt.use('Agg')
    d=history_feature_score
    plt.plot(d.columns,d.iloc[0],color="g")   
    plt.grid(True)
    plt.xlabel("Feature",fontsize=8)
    plt.ylabel("Accurancy" ,fontsize=8)
    plt.title(string+" Backward Elimination",fontsize=8)
    plt.savefig(string+"_Backward_Elimination.pdf", format="pdf")
    plt.close(None)
    return()
    
def backward_elimination(string,df, clf):
    print("BACKWARD_FEATURE_Elimination")
    startTime = time.time() 
    y=df[df.columns[len(df.columns)-1]]
    df=df.drop(df.columns[len(df.columns)-1], axis=1)
    scaler = StandardScaler()
    Xs=scaler.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1)
    feature_score=[]
    featname=list(df.columns)
    print(featname)
    print(clf.get_params)
    clf.fit(X_train,y_train)
    history_feature_score=pd.DataFrame([100*accuracy_score(y_test,clf.predict(X_test))],columns=["FullData"])
    print("Full Data Accuracy:", history_feature_score.iloc[0,0])
    while len(featname)>=2:
        featindx=list(range(len(featname)))
        for i in range(0,len(featname)):
            featindx.remove(i)
            clf.fit(X_train[:,featindx],y_train)
            y_pred=clf.predict(X_test[:,featindx])
            feature_score.append(100*accuracy_score(y_test, y_pred))
            featindx=list(range(len(featname)))
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            pd.options.display.float_format = "{:.14f}".format
            print(pd.DataFrame(np.array(feature_score),index=featname ).sort_values(by=[0], ascending=False) )
        maxindex=feature_score.index(max(feature_score))
        maxscoreback=feature_score[maxindex]
        history_feature_score[featname[maxindex]]=maxscoreback
        print("BACKWARD_FEATURE_Elimination_SCORE:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            pd.options.display.float_format = "{:.6f}".format
            print(history_feature_score)           
        featindx.remove(maxindex)
        X_train=X_train[:,featindx].copy()
        X_test=X_test[:,featindx].copy()
        featname.remove(featname[maxindex])
        if len(featname)==1:
            history_feature_score[featname[0]]=feature_score[0]
            print("BACKWARD_FEATURE_Elimination_SCORE:")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                pd.options.display.float_format = "{:.6f}".format
                print(history_feature_score)
            feature_score.clear()
            endTime = time.time()
            print("fit time")
            print(endTime - startTime)
            lineplot(history_feature_score, string)
            break
        feature_score.clear()
    return()

from sklearn.linear_model import LogisticRegression
string="Logistic_Regression"
clf=LogisticRegression( C=1e10,fit_intercept=False,  penalty="l2", solver="sag",max_iter=700)  
backward_elimination(string,df, clf)
 

    






