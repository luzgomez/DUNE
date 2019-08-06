import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, normaltest, levene, mannwhitneyu, fligner, ks_2samp
import matplotlib.pyplot as plt
import math
import sys
#df=pd.read_csv('/home/nestor/uk/21v/21v1k.txt',",", header=None)
df=pd.read_csv(sys.argv[1],",", header=None) 
feat=["1","2","3","4","5","6","7","8old","8new","9","10","11a","11b","12","13a","13b","13c","13d","14a","14b","TS" ]
alpha=0.01
rows=[]
for i in range(1, len(df.iloc[0])):
	rows.append(i)



def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step
 
distribution=[ 'anglit',  'arcsine', 'cauchy','cosine', 'expon', 'gilbrat', 'gumbel_r', 'halfcauchy', 'halflogistic',
              'halfnorm', 'hypsecant', 'laplace', 'logistic', 'maxwell', 'rayleigh', 'uniform', 'wald']



df=df.drop([0], axis=1)
df0=df[df[len(df.iloc[0])]==0]
df1=df[df[len(df.iloc[0])]==1]
df.columns=feat
df0.columns=feat
df1.columns=feat
n=len(df)
if n/1000000<1:
	if n/90000<1:
		nn=str(round(n/1000,2))+"k"
		b=10# divide bins
	else:
		nn=str(round(n/1000,2))+"k"
		b=800# divide bins
else:
	nn=str(round(n/1000000,2))+"M"
	b=2000# divide bins	

string="Number of samples="+str(n)
print(string)
print("----------------------------------------------------------------------------------------")
print("description of  total data")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.describe())
print("----------------------------------------------------------------------------------------")
print("description of data class 0")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	print(df0.describe())
print("----------------------------------------------------------------------------------------")
print("description of data class 1")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	print(df1.describe())
	print("----------------------------------------------------------------------------------------")
	print("Absolute value of diference of standarize means")
	print( (((df1.mean(axis = 0) -df0.mean(axis =0 ))/df.std(axis=0) ).abs()).sort_values(ascending=False)
 )

df.columns=rows
df0.columns=rows
df1.columns=rows

print("histograms of  total data")
for i in range(1,len(df.iloc[0])):
    fig = plt.figure(i)
    plt.hist(df[i],color='b', alpha=1, histtype='step', bins=int(n/b))
    plt.grid(True)
    plt.ylabel("Frecuency")
    plt.xlabel(" Feature  " +feat[i-1] ) 
    string="Histogram of Feature "+ feat[i-1] +" with "+ nn+ " samples"
    plt.title(string)
    string="Histogram_by_Feature"+ feat[i-1]+".pdf"
    #plt.show()
    fig.savefig(string,format="pdf")

print("----------------------------------------------------------------------------------------")
print("histograms of  variable separated by group")
col=['g','r'] 
for i in range(1,len(df.iloc[0])):
    fig = plt.figure(100+i)
    plt.hist(df0[i],  label='0',color=col[0], alpha=0.8, histtype='step',  bins=int(n/b))
    plt.hist(df1[i],  label='1',color=col[1], alpha=0.6, histtype='step',  bins=int(n/b))
    plt.grid(True)
    plt.ylabel("Frecuency")
    plt.xlabel(" Feature  " +feat[i-1] )
    plt.legend(loc='upper right')
    string="Histograms by group of Feature  " +feat[i-1] +" with "+ nn+ " samples"
    plt.title(string)
    string="Histogram_by_group_Feature" +feat[i-1]+".pdf"
    #plt.show()
    fig.savefig(string, format="pdf")
 
print("----------------------------------------------------------------------------------------")


#print("----------------------------------------------------------------------------------------")

pvmean=[]
pvdistribution=[]
pvnormal=[]
pvnormalclass0=[]
pvnormalclass1=[]
pvarianzalev=[]
pvarianzaflig=[]

## pruebas estadisticas
for i in range(1,len(df.iloc[0])):
    
    stat, p =normaltest(df[i])# normalidad de columna
    pvnormal.append(p)
    
    stat, p =normaltest(df0[i])# normalidad de columna clase 0
    pvnormalclass0.append(p)
    
    stat, p =normaltest(df1[i])# normalidad de columna clase 1
    pvnormalclass1.append(p)
    
    stat, p = mannwhitneyu(df0[i], df1[i])#prueba igualdad de equidistribucion
    pvdistribution.append(p)
    
    stat, p = ttest_ind(df0[i], df1[i])# medias diferentes
    pvmean.append(p)
    
    stat, p = levene(df0[i], df1[i])#prueba igualdad de varianza
    pvarianzalev.append(p)
    
    stat, p = fligner(df0[i], df1[i])#prueba igualdad de varianza
    pvarianzaflig.append(p)


#aclaraciones
#no se usa test de shapiro por que hay indicios que dicen que
# funciona mal con muchos datos

print("----------------------------------------------------------------------------------------")
print("D Agostino and Pearson s  test for each column, (testing normality)")
print("Pvalue")
print(pvnormal)
for j in range(1,len(df.iloc[0])):
    if pvnormal[j-1]>alpha:
        string="Normality for Feature "+feat[j-1]+" for full data with p value=" +str(pvnormal[j-1])
        print(string)

print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("D Agostino and Pearson s  test for each column class 0 (testing normality)")
print("Pvalue")
print(pvnormalclass0)
for j in range(1,len(df.iloc[0])):
    if pvnormalclass0[j-1]>alpha:
        string="Normality for Feature "+feat[j-1]+" for the data with class 0 and  p value=" +str(pvnormalclass0[j-1])
        print(string)
        
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("D Agostino and Pearson s  test for each column class 1 (testing normality)")
print("Pvalue")
print(pvnormalclass1)
for j in range(1,len(df.iloc[0])):
    if pvnormalclass1[j-1]>alpha:
        string="Normality for Feature "+feat[j-1]+" for the data with class 1 and  p value=" +str(pvnormalclass1[j-1])
        print(string)
        
print("----------------------------------------------------------------------------------------")
### !!!!!!!NADA ES NORMAL
print("----------------------------------------------------------------------------------------")
print("Mann Whitney U  test (equal distributed probability function between classes non parametric)")
print("Pvalue")
for j in range(1,len(df.iloc[0])):
    if pvdistribution[j-1]>alpha:
        string="Class probability distribution of Feature "+feat[j-1]+"  are the same with  p value=" +str(pvdistribution[j-1])
        print(string)
    else:
        string="Class probability distribution of Feature "+feat[j-1]+"  are the different with  p value=" +str(pvdistribution[j-1])
        print(string)
              
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("Levene test (equal variances non normal median center used)")
print("Pvalue")
for j in range(1,len(df.iloc[0])):
    if pvarianzalev[j-1]>alpha:
        string="Class variances of Feature "+feat[j-1]+"  are the same with  p value=" +str(pvarianzalev[j-1])
        print(string)
    else:
        string="Class  variances of Feature  "+feat[j-1]+"  are the different with  p value=" +str(pvarianzalev[j-1])
        print(string)
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("Fligner test (equal variances, non parametric)")
print("Pvalue")
for j in range(1,len(df.iloc[0])):
    if pvarianzaflig[j-1]>alpha:
        string="Class variances of Feature "+feat[j-1]+"  are the same with  p value=" +str(pvarianzaflig[j-1])
        print(string)
    else:
        string="Class  variances of Feature  "+feat[j-1]+"  are the different with  p value=" +str(pvarianzaflig[j-1])
        print(string)
print("----------------------------------------------------------------------------------------")
print("t test for column mean value  between groups (parametric) ")    
print("Pvalue")
for j in range(1,len(df.iloc[0])):
    if pvmean[j-1]>alpha:
        string="Class means of Feature "+feat[j-1]+"  are the same with  p value=" +str(pvmean[j-1])
        print(string)
    else:
        string="Class  means of Feature  "+feat[j-1]+"  are the different with  p value=" +str(pvmean[j-1])
        print(string)
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")

print("----------------------------------------------------------------------------------------")



for i in range(0, len(distribution)):

    for j in range(1,len(df.iloc[0])):
         stat, p=stats.kstest(df[j], distribution[i])

         if p>alpha:
             string=distribution[i]+" is the distribution of Feature "+feat[j-1]+" from the full data with  p value="+str(p)
             print(string)


print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("kolmogorov smirnov test distribution data class0")

for i in range(0, len(distribution)):

    for j in range(1,len(df.iloc[0])):
         stat, p=stats.kstest(df0[j], distribution[i])

         if p>alpha:
            string=distribution[i]+" is the distribution of Feature "+feat[j-1]+" from the data with class 0 with  p value="+str(p)
            print(string)


print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("kolmogorov smirnov test distribution data class1")


for i in range(0, len(distribution)):

    for j in range(1,len(df.iloc[0])):
         stat, p=stats.kstest(df1[j], distribution[i])

         if p>alpha:
             string=distribution[i]+" is the distribution of Feature "+feat[j-1]+" from the data with class 1 with  p value="+str(p)
             print(string)

print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")  
print("chi full data  ")
mu=df.mean()# mean full data
sig=df.var()# variance full data
for i in range (-2,3):

    for j in range(1,len(df.iloc[0])):
        k=math.ceil((mu[j]*mu[j])+sig[j])+i 
        if k>0:
            stat, p=stats.kstest(df[j], 'chi',[k,0,1])
            if p>alpha:
                string="chi_"+str(k)+"_is the distribution of Feature "+feat[j-1]+" from the full data with  p value="+str(p)
                print(string)

    
print("----------------------------------------------------------------------------------------")  
print("chi class 0 data ")
mu0=df0.mean()# mean class 0 data
sig0=df0.var()# variance class 0 data
for i in range (-2,3):

    for j in range(1,len(df.iloc[0])):
        k=math.ceil((mu0[j]*mu0[j])+sig0[j])+i
        if k>0:
           stat, p=stats.kstest(df0[j], 'chi',[k,0,1])
           if p>alpha:
                string="chi_"+str(k)+"_is the distribution of Feature"+feat[j-1]+" from the data with class 0 with  p value="+str(p)
                print(string)

print("----------------------------------------------------------------------------------------")  
print("chi class 1 data ")
mu1=df1.mean()# mean class 1 data
sig1=df1.var()#variance class 1 data
for i in range (-2,3):

    for j in range(1,len(df.iloc[0])):
        k=math.ceil((mu1[j]*mu1[j])+sig1[j])+i
        if k>0:
               stat, p=stats.kstest(df1[j], 'chi',[k,0,1])
               if p>alpha:
                   string="chi_"+str(k)+"_is the distribution of Feature"+feat[j-1]+" from the data with class 1 with  p value="+str(p)
                   print(string)

print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")  
print("chi2 full data ")
for i in range (-2,3):

    for j in range(1,len(df.iloc[0])):
        k=math.ceil(mu[j])+i
        if k>0:
            stat, p=stats.kstest(df[j], 'chi2',[k,0,1])
            if p>alpha:
                string="chi2_"+str(k)+"_is the distribution of Feature "+feat[j-1]+" from the full data with  p value="+str(p)
                print(string)

print("----------------------------------------------------------------------------------------")  
print("chi2 class 0 data")
for i in range (-2,3):

    for j in range(1,len(df.iloc[0])):
        k=math.ceil(mu0[j])+i
        if k>0:
            stat, p=stats.kstest(df0[j], 'chi2',[k,0,1])
            if p>alpha:
                string="chi2_"+str(k)+"_is the distribution of Feature"+feat[j-1]+" from the data with class 0 with  p value="+str(p)
                print(string)

    
print("----------------------------------------------------------------------------------------")  
print("chi2 class 1 data")
for i in range (-2,3):

    for j in range(1,len(df.iloc[0])):
        k=math.ceil(mu1[j])+i
        if k>0:
            stat, p=stats.kstest(df1[j], 'chi2',[k,0,1])
            if p>alpha:
                   string="chi2_"+str(k)+"_is the distribution of Feature"+feat[j-1]+" from the data with class 1 with  p value="+str(p)
                   print(string)

print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")  
print("----------------------------------------------------------------------------------------")  
print("gamma full data ")
for a in frange(-0.9, 1, 0.1):
    for b in frange (-0.9, 1, 0.1):

        for j in range(1,len(df.iloc[0])):
            k=(mu[j]*mu[j])/sig[j]+a
            s=(mu[j])/sig[j]+b
            if k>0 and s>0:
                stat, p=stats.kstest(df[j], 'gamma',[k,0,s])
                if p>alpha:
                    string="gamma_K="+str(round(k,2))+" Theta="+str(round(s),2)+"is the distribution of Feature "+feat[j-1]+" from the full data with  p value="+str(p)
                    print(string)

print("----------------------------------------------------------------------------------------")  
print("gamma class 0 data")
for a in frange(-0.9, 1, 0.1):
    for b in frange (-0.9, 1, 0.1):

        for j in range(1,len(df.iloc[0])):
            k=(mu0[j]*mu0[j])/sig0[j]
            s=(mu0[j])/sig0[j]
            if k>0 and s>0:
                stat, p=stats.kstest(df0[j], 'gamma',[k,0,s])
                if p>alpha:
                    string="chi2_K="+str(round(k,2))+" Theta="+str(round(s),2)+"is the distribution of Feature "+feat[j-1]+" from the data with class 0 with  p value="+str(p)
                    print(string)

print("----------------------------------------------------------------------------------------")  
print("gamma class 1 data")
for a in frange(-0.9, 1, 0.1):
    for b in frange (-0.9, 1, 0.1):

        for j in range(1,len(df.iloc[0])):
            k=(mu1[j]*mu1[j])/sig1[j]
            s=(mu1[j])/sig1[j]
            if k>0 and s>0:
                stat, p=stats.kstest(df1[j], 'gamma',[k,0,s])
                if p>alpha:
                    string="chi2_K="+str(round(k,2))+" Theta="+str(round(s),2)+"is the distribution of Feature "+feat[j-1]+" from the data with class 1 with  p value="+str(p)
                    print(string)

#print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("exponencial full data ")
for k in frange(-0.9, 1, 0.1):

    for j in range(1,len(df.iloc[0])):
        s=k+mu[j]
        if s>0:
             stat, p=stats.kstest(df[j],'expon',[0,s])
             if p>alpha:
                 string="expo_l="+str(round(1/s,2))+" is the distribution of Feature "+feat[j-1]+" from the full data with  p value="+str(p)
                 print(string)

print("----------------------------------------------------------------------------------------")  
print("exponencial class 0 ")
for k in frange(-0.9, 1, 0.1):

    for j in range(1,len(df.iloc[0])):
        s=k+mu0[j]
        if s>0:
            stat, p=stats.kstest(df0[j], 'expon',[0,s])
            if p>alpha:
                 string="expo_l="+str(round(1/s,2))+" is the distribution of Feature "+feat[j-1]+" from the data with class 0 with  p value="+str(p)
                 print(string)

print("----------------------------------------------------------------------------------------")  
print("exponencial class 1 data ")
for k in frange(-0.9, 1, 0.1):

        for j in range(1,len(df.iloc[0])):
            s=k+mu1[j]
            if s>0:
                stat, p=stats.kstest(df1[j],'expon',[0,s])
                if p>alpha:
                 string="expo_l="+str(round(1/s,2))+" is the distribution of Feature "+feat[j-1]+" from the data with class 1 with  p value="+str(p)
                 print(string)

print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")  
print("----------------------------------------------------------------------------------------")  
print("----------------------------------------------------------------------------------------")  
print("Test for equidistribution between Features  in the full data Mann Whitney U")
print(" ")
mannequidistribution=pd.DataFrame()
for i in range(1,len(df.iloc[0])+1):
    vectdistribution=[]
    for j in range(1,len(df.iloc[0])+1):
        if (i<=j):
            stat, p = mannwhitneyu(df[i], df[j])#prueba igualdad de equidistribucion
            vectdistribution.append(p)
            if p>alpha:
                string="Features "+feat[i-1]+","+feat[j-1]+" comes from the same distribution (full data) with a pvalue= "+ str(p)
                print(string)
        else:
             vectdistribution.append(-1)
    mannequidistribution[i]=vectdistribution
print(" ")
mannequidistribution.columns=feat
mannequidistribution.index=feat
print("Full matrix pvalue of equidistribution test")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	print(mannequidistribution)

print("----------------------------------------------------------------------------------------")  
print("----------------------------------------------------------------------------------------")

print("Test for equidistribution between Features  in the full data Kolmogorov Smirnov")
print(" ")
ks2sampequidistribution=pd.DataFrame()
for i in range(1,len(df.iloc[0])+1):
    vectdistribution=[]
    for j in range(1,len(df.iloc[0])+1):
        if (i<=j):
            stat, p = ks_2samp(df[i], df[j])#prueba igualdad de equidistribucion
            vectdistribution.append(p)
            if p>alpha:
                string="Features "+feat[i-1]+","+feat[j-1]+" comes from the same distribution (full data) with a pvalue= "+ str(p)
                print(string)
        else:
             vectdistribution.append(-1)
    ks2sampequidistribution[i]=vectdistribution
print(" ")
ks2sampequidistribution.columns=feat
ks2sampequidistribution.index=feat
print("Full matrix pvalue of equidistribution Kolmogorov Smirnov test")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	print(ks2sampequidistribution)

print("----------------------------------------------------------------------------------------")  
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("correlation matrix full data")
corre=df.corr(method='pearson')

print(" ")
for i in range(1,len(df.iloc[0])+1):
    for j in range(1,len(df.iloc[0])+1):
        if (i<j):
            if corre[i][j]>=0.8 or corre[i][j]<=-0.8:
                string="Correlation between Feature "+feat[i-1]+","+feat[j-1]+" is "+ str(corre[i][j])
                print(string)
corre.columns=feat
corre.index=feat
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	print(corre)            
            
print("----------------------------------------------------------------------------------------")  
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")

print("correlation matrix with class 0 ")
corre0=df0.corr(method='pearson')

print(" ")
for i in range(1,len(df.iloc[0])+1):
    for j in range(1,len(df.iloc[0])+1):
        if (i<j):
            if corre0[i][j]>=0.8 or corre0[i][j]<=-0.8:
                string="Correlation between Feature "+feat[i-1]+","+feat[j-1]+" is "+ str(corre0[i][j])
                print(string)

corre0.columns=feat
corre0.index=feat
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	print(corre0) 
print("----------------------------------------------------------------------------------------")  
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("correlation matrix with class 1")
corre1=df1.corr(method='pearson')

print(" ")
for i in range(1,len(df.iloc[0])+1):
    for j in range(1,len(df.iloc[0])+1):
        if (i<j):
            if corre1[i][j]>=0.8 or corre1[i][j]<=-0.8:
                string="Correlation between Feature "+feat[i-1]+","+feat[j-1]+" is "+ str(corre1[i][j])
                print(string)

corre1.columns=feat
corre1.index=feat
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	print(corre1) 
print("----------------------------------------------------------------------------------------")
