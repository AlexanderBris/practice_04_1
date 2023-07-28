# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:21:01 2023

@author: Alexander
"""

# подгрузка стандартных библиотек
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# функция чтения данных из файла
def read_data(path, filename):
    return pd.read_csv(os.path.join(path, filename))

# загружаем данные в переменную df
cur_dir = os.getcwd()
# файлы лежат в папке с основным скриптом
dataset = read_data(cur_dir, 'airline-passengers.csv')
# проверка
dataset.head()

# просмотр данных
#-----------------------------------------------------------------------------

training_data = dataset.iloc[:, 1:2].values # transform dataframe to numpy.array
# plotting
plt.figure(figsize=(12, 4))
plt.plot(training_data, label='Airline Passangers Data')
plt.title("Number of passengers per month")
plt.ylabel("#passengers")
plt.xlabel("Month")
labels_to_display = [i for i in range(training_data.shape[0]) if i % 12 == 0]
plt.xticks(labels_to_display, dataset['Month'][labels_to_display])
plt.grid()
plt.show()

# масштабирование данных
#-----------------------------------------------------------------------------
td_min = training_data.min()
td_max = training_data.max()
print('Initial statistics:')
print('Minimum value:', repr(td_min).rjust(5))
print('Maximum value:', repr(td_max).rjust(5))

training_data = (training_data - td_min) / (td_max - td_min)
print('\nResulting statistics:')
print('Minimum value:', repr(training_data.min()).rjust(5))
print('Maximum value:', repr(training_data.max()).rjust(5))

# разбиение данных
#-----------------------------------------------------------------------------
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length):
        _x = data[i:(i + seq_length)] # picking several sequential observations
        _y = data[i+seq_length] # picking the subsequent observation
        x.append(_x)
        y.append(_y)

    return torch.Tensor(np.array(x)), torch.Tensor(np.array(y))
    
# set length of the ensemble; accuracy of the predictions and 
# speed perfomance almost always depend on it size
seq_length = 8 # compare 2 and 32
x, y = sliding_windows(training_data, seq_length)
print("Example of the obtained data:\n")
print("Data corresponding to the first x:")
print(x[0])
print("Data corresponding to the first y:")
print(y[0])

train_size = int(len(y) * 0.8)

x_train = x[:train_size]
y_train = y[:train_size]
#print(x_train)

x_test = x[train_size:]
y_test = y[train_size:]

print("Train data:")
print("x shape:", x_train.shape)
print("y shape:", y_train.shape)

print("\nTest data:")
print("x shape:", x_test.shape)
print("y shape:", y_test.shape)

# создание и обучение
#-----------------------------------------------------------------------------
class AirTrafficPredictor(nn.Module):

    def __init__(self, input_size, hidden_size):
        # hidden_size == number of neurons 
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # Predict only one value

    def forward(self, x):
        #print("x: ",x.shape) # 108 x 8 x 1 : [batch_size, seq_len, input_size] 
        out, h = self.rnn(x) 
       #print("out: ", out.shape) # 108 x 8 x 4 : [batch_size, seq_len, hidden_size] Useless!
        #print("h : ", h.shape) # 1 x 108 x 4 [ num_layers, batch_size, hidden_size]
        y = self.fc(h)
        #print("y",y.shape) # 1 x 108 x 1
        return y, h

def time_series_train(model, num_epochs=2000, learning_rate=0.01):
  
  criterion = torch.nn.MSELoss() # mean-squared error for regression
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Train the model
  for epoch in range(num_epochs):
      y_pred, h = model(x_train) # we don't use h there, but we can!
      optimizer.zero_grad()
      
      # obtain the loss
      loss = criterion(y_pred[0], y_train) # for shape compatibility
      loss.backward()
      
      optimizer.step()
      if epoch % 100 == 0:
          print(f"Epoch: {epoch},".ljust(15), "loss: %1.5f" % (loss.item()))

print("Simple GRU training process with MSE loss:")
input_size = 1
hidden_size = 4   #количество слоев в GRU блок
rnn = AirTrafficPredictor(input_size, hidden_size)
time_series_train(rnn)

# проверка
#-----------------------------------------------------------------------------
def time_series_plot(train_predict):
  data_predict = train_predict.data
  y_data_plot = y.data

  # Denormalize
  data_predict = data_predict[0] * (td_max - td_min) + td_min
  y_data_plot = y_data_plot * (td_max - td_min) + td_min 

  # Plotting
  plt.figure(figsize=(12, 4))
  plt.axvline(x=train_size, c='r', linestyle='--')
  # shifting the curve as first y-value not correspond first value overall
  plt.plot(seq_length + np.arange(y_data_plot.shape[0]), y_data_plot)
  plt.plot(seq_length + np.arange(y_data_plot.shape[0]), data_predict)
  
  plt.title("Number of passengers per month")
  plt.ylabel("#passengers")
  plt.xlabel("Month")
  plt.xticks(labels_to_display, dataset['Month'][labels_to_display])
  
  plt.legend(['Train/Test separation', 'Real', 'Predicted'])
  plt.grid(axis='x')
  plt.show()

rnn.eval()
train_predict, h = rnn(x)
time_series_plot(train_predict)
