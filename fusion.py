import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import LSTM,Dropout,GRU,MaxPooling1D,Flatten
from keras.models import Sequential
from keras.layers import Dense,Concatenate,Bidirectional
from keras.layers.convolutional import Conv1D
import tensorflow as tf
from keras.utils.vis_utils import model_to_dot
from keras.layers import RepeatVector
from keras.layers import TimeDistributed,Activation
from sklearn.preprocessing import MinMaxScaler
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


cols_normalize = FD1_train.columns.difference(
    ['unit_number', 'time_in_cycles', 'RUL', 'label1', 'label2'])  # NORMALIZE COLUMNS except [id , cycle, rul ....]

min_max_scaler = MinMaxScaler()

norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(FD1_train[cols_normalize]),
                             columns=cols_normalize,
                             index=FD1_train.index)

join_df = FD1_train[FD1_train.columns.difference(cols_normalize)].join(norm_train_df)
FD1_train = join_df.reindex(columns=FD1_train.columns)
print(FD1_train)

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
y, z = train_x[:,0:62,:], train_x[:,62:127:]
y1, z1 = train_y[0:62], train_y[62:127]

model = Sequential()
model.add(Conv1D(128, 1, activation="relu", input_shape=(62, 15)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, 1, activation="relu", input_shape=(62, 15)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, 1, activation="relu", input_shape=(62, 15)))
model.add(MaxPooling1D(pool_size=2))


model1=Sequential()
model1.add(LSTM(128, input_shape=(65, 15), return_sequences=True))
model1.add(LSTM(64, input_shape=(65, 15), return_sequences=True))
model1.add(LSTM(32, input_shape=(65, 15), return_sequences=False))

result=Sequential()
a=Concatenate([model, model1])
result.add(a)
result.add(Conv1D(128, 1, activation="relu", input_shape=(127, 15)))
result.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, kernel_initializer='uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit([y,z],[y1,z1] ,batch_size=200,epochs=10,validation_split=0.10,verbose=1)



