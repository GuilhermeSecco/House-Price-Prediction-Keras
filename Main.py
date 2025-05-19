import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

dataset = pd.read_csv('kc_house_data.csv')

X = dataset.iloc[:, 3:].values
x = X[:, np.r_[0:13,14,18:]]
y = dataset.iloc[:,2].values

xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size= 0.2,random_state=0)

xscaler = MinMaxScaler(feature_range=(0,1))
xTrain = xscaler.fit_transform(xTrain)
xTest = xscaler.transform(xTest)

yscaler = MinMaxScaler(feature_range=(0,1))
yTrain = yscaler.fit_transform(yTrain.reshape(-1,1))
yTest = yscaler.transform(yTest.reshape(-1,1))

model = Sequential()
model.add(Dense(units = 64, activation = 'linear', input_dim = x.shape[1]))
model.add(Dense(units = 16, activation = 'linear'))
model.add(Dense(units = 1, activation = 'linear'))
model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'mse', metrics = ['mean_absolute_error'])

model.fit(xTrain, yTrain, batch_size = 32, epochs = 100, validation_data = (xTest,yTest))

yTest = yscaler.inverse_transform(yTest)
prediction = yscaler.inverse_transform(model.predict(xTest))

error = abs(prediction - yTest)/yTest
print(np.mean(error))