import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mlt
from sklearn.metrics import  accuracy_score
import sys
import time

#df=pd.read_csv(sys.argv[1],",", header=None)
df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
df=df.drop([0], axis=1)
df.columns=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df=df.drop("14a", axis=1)
df=df.drop("11b", axis=1)
select=["8old","8new","11a","12","13a","13b","13c","13d","14b"]
#select=["1","2","3","5","6","7","8old","8new","9","10","11a","12","13a","13b","13c","13d","14b"]

def lineplot(history_feature_score, string):
    plt.rcParams.update({'font.size': 6})
    plt.rcParams['lines.linewidth'] = 0.8
    mlt.use('Agg')
    d=history_feature_score
    plt.plot(d.columns,d.iloc[0],color="g")   
    plt.grid(True)
    plt.xlabel("Feature",fontsize=8)
    plt.ylabel("Accurancy" ,fontsize=8)
    plt.title(string+" Forward Selection",fontsize=8)
    plt.savefig(string+"_Forward_Selection.pdf", format="pdf")
    plt.close(None)
    return()

def fordward_main(string,df, model):
    y=df[df.columns[len(df.columns)-1]]
    df=df.drop(df.columns[len(df.columns)-1], axis=1)
    scaler = StandardScaler()
    Xs=scaler.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1)
    print ("testsize=0.25")
    X_train=pd.DataFrame(X_train)
    X_test=pd.DataFrame(X_test)
    X_test.columns=df.columns
    X_train.columns=df.columns
    selection_train=X_train[select].copy()
    selection_test=X_test[select].copy()
    X_test.drop(select, axis=1, inplace=True)
    X_train.drop(select, axis=1, inplace=True)
    feature_score=[]
    print(select)
    print(clf.get_params)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    history_feature_score=pd.DataFrame([100*accuracy_score(y_test, y_pred)],columns=["MainData"])
    print("Main Data Accuracy:", history_feature_score.iloc[0,0])
    startTime = time.time() 
    print("FORWARD_Main_FEATURE_SELECTION")
    while len(select)>0:
        for i in select:
            X_train[i]=selection_train[i]
            X_test[i]=selection_test[i]
            clf.fit(X_train.values,y_train)
            y_pred=clf.predict(X_test.values)
            feature_score.append(100*accuracy_score(y_test, y_pred))
            X_test.drop(i, axis=1, inplace=True)
            X_train.drop(i, axis=1, inplace=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
            pd.options.display.float_format = "{:.14f}".format
            print(pd.DataFrame(np.array(feature_score),index=select).sort_values(by=[0], ascending=False) )    
        maxindex=feature_score.index(max(feature_score))
        maxscoreford=feature_score[maxindex]        
        history_feature_score[select[maxindex]]=maxscoreford
        print("FORWARD_Main_FEATURE_SELECTION_SCORE:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            pd.options.display.float_format = "{:.6f}".format
            print(history_feature_score)
        X_train[select[maxindex]]=selection_train[select[maxindex]]
        X_test[select[maxindex]]=selection_test[select[maxindex]]   
        selection_train.drop(select[maxindex], axis=1, inplace=True)
        selection_test.drop(select[maxindex], axis=1, inplace=True)
        select.remove(select[maxindex])
        if len(select)==0:
            break
        feature_score.clear()
    endTime = time.time()
    print("fit time")
    print(endTime - startTime) 
    lineplot(history_feature_score, string)
    return()

from sklearn.linear_model import LogisticRegression
string="Logistic_Regression"
clf=LogisticRegression( C=1e10,fit_intercept=False,  penalty="l2", solver="sag",max_iter=700) 
fordward_main(string,df, clf)
    