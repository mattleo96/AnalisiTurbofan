import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dropout,GRU,MaxPooling1D,Flatten,Conv1D
from keras.models import Sequential
from keras.layers import Dense,Concatenate,Bidirectional
from sklearn.preprocessing import QuantileTransformer , PowerTransformer
from keras import layers,Model,Input
from keras.layers import concatenate
import tensorflow as tf
df = pd.read_csv("/Users/Mattia/Desktop/CMaps/train_FD003.txt",delimiter=" ",header = None)
test_file = pd.read_csv("/Users/Mattia/Desktop/CMaps/train_FD001.txt",delimiter=" ",header = None)
RUL_file=pd.read_csv("/Users/Mattia/Desktop/CMaps/RUL_FD001.txt",header = None)

df.drop(columns=[26,27],inplace=True)
columns = ["Section-{}".format(i)  for i in range(26)]
df.columns = columns
MachineID_name = ["Section-0"]
RUL_name = ["Section-1"]
OS_name = ["Section-{}".format(i) for i in range(2,5)]
Sensor_name = ["Section-{}".format(i) for i in range(5,26)]

# Data in pandas DataFrame
MachineID_data = df[MachineID_name]
RUL_data = df[RUL_name]
OS_data = df[OS_name]
Sensor_data = df[Sensor_name]

# Data in pandas Series
MachineID_series = df["Section-0"]
RUL_series = df["Section-1"]
grp = RUL_data.groupby(MachineID_series)
max_cycles = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()])
df.drop(columns=["Section-0",
                "Section-4", # Operatinal Setting
                "Section-5", # Sensor data
                "Section-9", # Sensor data
                "Section-10", # Sensor data
                "Section-14",# Sensor data
                "Section-20",# Sensor data
                "Section-22",# Sensor data
                "Section-23"] , inplace=True)
gen = MinMaxScaler(feature_range=(0, 1))
df = gen.fit_transform(df)
df = pd.DataFrame(df)
#df = df.rolling(20).mean()
pt = PowerTransformer()
df = pt.fit_transform(df)
def RUL_df():
    rul_lst = [j  for i in MachineID_series.unique() for j in np.array(grp.get_group(i)[::-1]["Section-1"])]
    rul_col = pd.DataFrame({"rul":rul_lst})
    return rul_col
X_train = np.array(df)
y_train = np.array(RUL_df()).reshape(-1,1)
print(X_train.shape)
print(y_train.shape)



def create_dataset(X, look_back=10):
    data = []
    for i in range(len(X)-look_back-1):
        data.append(X[i:(i+look_back)])
    return np.array(data)

X_train = create_dataset(X_train)
y_train = y_train[11:]

def build_model(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    #model.add(Dropout(d))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = [tf.keras.metrics.RootMeanSquaredError()])
    return model
model = build_model([17,10])

history = model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=75,
    validation_split=0.15,
    verbose=1)


