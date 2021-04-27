import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
FD1_train = pd.read_csv("/Users/Mattia/Desktop/CMaps/train_FD003.txt",delimiter=" ",header = None)
FD1_test = pd.read_csv("/Users/Mattia/Desktop/CMaps/train_FD001.txt",delimiter=" ",header = None)
RUL=pd.read_csv("/Users/Mattia/Desktop/CMaps/RUL_FD001.txt",header = None)
def Calcolo_RUL(df):
    df_copy=df.copy()
    gruppo=df_copy.groupby('unit_number')['time_in_cycles'].max().reset_index()
    gruppo.columns = ['unit_number','max_cycles_fu']
    df_copy = df_copy.merge(gruppo, on=['unit_number'], how='left')
    df['RUL'] = df_copy['max_cycles_fu'] - df_copy['time_in_cycles']
def screen_data(df):
    print('-'*40)
    print('Valori mancanti :')
    print((df.isnull().sum()/df.shape[0])*100)
    print('-'*40)
    print('Valori unici:')
    print(df.nunique())
    print('-'*40)
FD1_train=FD1_train.drop([26, 27], axis = 1)
columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
FD1_train.columns=columns

screen_data(FD1_train)


## Plot valori sensori
a=FD1_train.drop(['unit_number','time_in_cycles'], axis = 1)
a.plot(subplots =True, sharex = True, figsize = (30,30),layout=(13,2))
plt.tight_layout()
plt.show()

## studio lunghezza sequenze 
a=FD1_train.groupby('unit_number')['time_in_cycles'].max().reset_index()
print("Lunghezza massima sequenza:",FD1_train['time_in_cycles'].max())
print("Lunghezza media sequenza",a['time_in_cycles'].mean())
print("Lunghezza minimaa sequenza",a['time_in_cycles'].min())


## P_value per studio siginificatività
Calcolo_RUL(FD1_train)
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('RUL ~ setting_1+setting_2+TRA+T2+T24+T30+T50+P2+P15+P30+Nf+Nc+epr+Ps30+phi+NRf+NRc+BPR+farB+htBleed+Nf_dmd+PCNfR_dmd+W31+W32',data=FD1_train).fit()
aov_table = sm.stats.anova_lm(model)
p_value=round(aov_table['PR(>F)'],3).sort_values(ascending=False)
#print(p_value)
#Raccolgo feautere più significative 0.05
F_significative=p_value[p_value>0.05]
print(F_significative)





