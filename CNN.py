import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, LSTM,Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
import tensorflow as tf
from keras.layers import RepeatVector
from keras.layers import TimeDistributed,Activation
FD1_train = pd.read_csv("/Users/Mattia/Desktop/CMaps/train_FD001.txt",delimiter=" ",header = None)
FD1_test = pd.read_csv("/Users/Mattia/Desktop/CMaps/train_FD001.txt",delimiter=" ",header = None)
RUL=pd.read_csv("/Users/Mattia/Desktop/CMaps/RUL_FD001.txt",header = None)
FD1_train=FD1_train.drop([26, 27], axis = 1)
columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
FD1_train.columns=columns
FD1_train=FD1_train.drop(columns = ["Nf_dmd","PCNfR_dmd","TRA","P2","T2","farB","epr","P15","setting_2"], axis =1)
def Calcolo_RUL(df):
    df_copy=df.copy()
    gruppo=df_copy.groupby('unit_number')['time_in_cycles'].max().reset_index()
    gruppo.columns = ['unit_number','max_cycles_fu']
    df_copy = df_copy.merge(gruppo, on=['unit_number'], how='left')
    df['RUL'] = df_copy['max_cycles_fu'] - df_copy['time_in_cycles']
Calcolo_RUL(FD1_train)
data_cols = FD1_train.drop(['unit_number','time_in_cycles', "RUL"], axis=1).columns

def prepare_train_data(df):
    data_list = []
    target_list = []
    for unit_number in df.unit_number.unique():
        unit = df[df.unit_number == unit_number]
        data_list.append(np.array(unit[data_cols])[:127,:])
        target_list.append(np.array(unit["RUL"])[127])
    return (np.stack(data_list), np.array(target_list).T)
train_x, train_y = prepare_train_data(FD1_train)

model = Sequential()
model.add(Conv1D(64, 1, activation="relu", input_shape=(127, 15)))
model.add(LSTM(128, input_shape=(127, 15), return_sequences=True))
model.add(Dropout(0,2))
model.add(LSTM(64, input_shape=(127, 15), return_sequences=False))
model.add(Dropout(0,2))
model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
model.compile(loss='mean_squared_error', optimizer='RMSprop')
model.fit(train_x,train_y ,batch_size=512,epochs=200,validation_split=0.30,verbose=1)
print(model.evaluate(train_x,train_y))
"""
model = Sequential()
model.add(Conv1D(64, 1, activation="relu", input_shape=(127, 15)))
model.add(LSTM(64,return_sequences=True))
model.add(LSTM(32,return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='RMSprop')
history = model.fit(train_x, train_y, validation_split=0.33, epochs=10, batch_size=30)

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""