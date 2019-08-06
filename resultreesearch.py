import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('/home/nestor/uk/21v/trees/treesearch75percent.csv',",", header=None)
#df=df.sort_values(by=["mean_test_score"], ascending=False)
df=df.drop([0], axis=1)
df.columns=df.iloc[0,:]
df=df.drop([0], axis=0)
df=df.convert_objects(convert_numeric=True)
df['mean_test_score']=df['mean_test_score']*100

dfe=df[df['param_criterion']=="entropy"]
dfg=df[df['param_criterion']=="gini"]
dfe=dfe.drop(['mean_fit_time', 'param_criterion', 'mean_train_score'], axis=1)
dfg=dfg.drop(['mean_fit_time', 'param_criterion', 'mean_train_score'], axis=1)
dfe=dfe.sort_values(by=["mean_test_score"], ascending=False)
dfg=dfg.sort_values(by=["mean_test_score"], ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("entropy")
    print(dfe)
    print("gini")
    print(dfg)
import seaborn as sns
fig, axs = plt.subplots(ncols=2)
dfe = dfe.pivot('param_max_depth', 'param_max_features', 'mean_test_score')
sns.heatmap(dfe,annot=True,ax=axs[0],fmt='g',cmap="BuGn_r").set_title('entropy tree')
dfg = dfg.pivot('param_max_depth', 'param_max_features', 'mean_test_score')
sns.heatmap(dfg,annot=True,ax=axs[1],fmt='g',cmap="BuGn_r").set_title('gini tree')
#
