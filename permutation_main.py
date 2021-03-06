import pandas as pd
import numpy as np
from statistics import stdev, mean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mlt
import time
import sys

#df=pd.read_csv(sys.argv[1],",", header=None)
df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
df=df.drop([0], axis=1)
df.columns=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
df=df.drop("14a", axis=1)
df=df.drop("11b", axis=1)

    
def lineplot(history_feature_score, string):
    plt.rcParams.update({'font.size': 6})
    plt.rcParams['lines.linewidth'] = 0.8
    mlt.use('Agg')
    d=history_feature_score
    plt.plot(d.columns,d.iloc[0])   
    plt.grid(True)
    plt.xlabel("Feature",fontsize=8)
    plt.ylabel("Accurancy" ,fontsize=8)
    plt.title(string+" Backward Permutation Main Feature Elimination",fontsize=8)
    plt.savefig(string+"_Backward_Permutation_Main_Feature_Elimination.pdf", format="pdf")
    plt.close(None)
    return()
    
def backward_permutation_main_selection(string,df, clf, nperm):
    print("BACKWARD_PERMUTATION_FEATURE_Elimination")
    select=["8old","8new","11a","12","13a","13b","13c","13d","14b"]
    #select=["1","2","3","4","5","6","7","8old","8new","9","10","11a","12","13a","13b","13c","13d","14b"]
    y=df[df.columns[len(df.columns)-1]]
    df=df.drop(df.columns[len(df.columns)-1], axis=1)
    scaler = StandardScaler()
    Xs=scaler.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1)
    X_shuffle=np.copy(X_test)
    startTime = time.time()
    feature_score=[]
    sd_feature_score=[]
    CImin=[]
    CImax=[]
    E=(1.96/(nperm**0.5))
    featname=list(df.columns)
    print(clf.get_params)
    clf.fit(X_train,y_train)
    basescore=100-100*accuracy_score(y_test,clf.predict(X_test))
    history_feature_score=pd.DataFrame([100-basescore],columns=["FullData"])
    print("Full Data Accuracy:", history_feature_score.iloc[0,0])
    number=[]
    while len(select)>0:
        number.clear()
        for s in select:
            number.append(featname.index(s))    
        for i in number:
            npermscore=[] 
            for j in range(nperm):
                X_shuffle[i]=np.random.permutation(X_shuffle[i])
                y_pred=clf.predict(X_shuffle)
                npermscore.append(100-100*accuracy_score(y_test, y_pred))
                X_shuffle[:,i]=np.copy(X_test[:,i])              
            feature_score.append(mean(npermscore/basescore))  
            sd_feature_score.append(stdev(npermscore/basescore)) 
            CImin.append(feature_score[-1]-(E*sd_feature_score[-1]))
            CImax.append(feature_score[-1]+(E*sd_feature_score[-1]))
            Error_permu=100-np.asarray(feature_score)*basescore
        minindex=feature_score.index(min(feature_score))
        varimport=pd.DataFrame({"Mean Error Ratio":feature_score,"SD Error Ratio":sd_feature_score,"[  ":CImin,"]-95%":CImax,"Permutation Error":Error_permu },index=select)
        print("Permutation Score")
        varimport=varimport.sort_values(by=["Mean Error Ratio"], ascending=False)
        print(varimport) 
        if len(select)==1 and len(featname)==1:
            clf.fit(X_train,y_train)
            basescore=100-100*accuracy_score(y_test,clf.predict(X_test))   
            history_feature_score[featname[number[minindex]]]=100-basescore
            print("BACKWARD_PERMUTATION_ Main_FEATURE_Elimination_SCORE:")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(history_feature_score)
            lineplot(history_feature_score, string)
            endTime = time.time()
            print("fit time")
            print(endTime - startTime)
            break
        featindx=list(range(len(featname)))
        featindx.remove(number[minindex])
        X_shuffle=X_shuffle[:,featindx].copy()
        X_train=X_train[:,featindx].copy()
        X_test=X_test[:,featindx].copy()
        clf.fit(X_train,y_train)
        basescore=100-100*accuracy_score(y_test,clf.predict(X_test))   
        history_feature_score[featname[number[minindex]]]=100-basescore
        print("BACKWARD_PERMUTATION_ Main_FEATURE_Elimination_SCORE:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(history_feature_score)
        featname.remove(featname[number[minindex]])
        select.remove(select[minindex])
        if len(select)==0 and len(featname)!=0:
            lineplot(history_feature_score, string)
            endTime = time.time()
            print("fit time")
            print(endTime - startTime)
            break
        feature_score.clear()
        sd_feature_score.clear()
        CImin.clear()
        CImax.clear()
    print(" ########################################################################")
    return()


nperm=2
from sklearn.linear_model import LogisticRegression
string="Logistic_Regression"
clf=LogisticRegression( C=1e10,fit_intercept=False,  penalty="l2", solver="sag",max_iter=700) 
backward_permutation_main_selection(string,df, clf, nperm)