# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:57:10 2022

@author: pc
"""

import os 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules_for_stock_price import EDA, ModelCreation, Model_Evaluation

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

from tensorflow.keras.utils import plot_model

#%% Statics
DATA_PATH = os.path.join(os.getcwd(),'Datasets', 'Top_Glove_Stock_Price_Train.csv')
MMS_FILE_NAME = os.path.join(os.getcwd(),'minmax_scaler.pkl')

#%% Data Loading 
df = pd.read_csv(DATA_PATH)

#%% Data Inspection 

df.info()
df.describe().T

eda = EDA()
eda.plot_graph(df)

#%% Data Cleaning 

df['Close'] = pd.to_numeric(df['Close'],errors='coerce')
df.info()
df.isna().sum() # no NaNs value


#%% Preprocessing

mms = MinMaxScaler()
df = mms.fit_transform(np.expand_dims(df['Open'],axis=-1))

x_train = [] # initialise empty list
y_train = []

win_size = 60 # constant

for i in range(win_size,np.shape(df)[0]): # to get range inside from 60 till last
    x_train.append(df[i-win_size:i,0])
    y_train.append(df[i,0])
    
x_train = np.array(x_train)
y_train = np.array(y_train) 

#%% Model development (train_data)

mc = ModelCreation()
model = mc.simple_lstm_layer(x_train,num_node=128)


model.compile(loss='mse',optimizer='adam',
              metrics='mape')
x_train = np.expand_dims(x_train,axis=-1)
hist = model.fit(x_train,y_train,epochs=100,
          batch_size=32)

plot_model(model,show_layer_names=(True),show_shapes=(True))

#%% Model evaluation

hist.history.keys()

hist_me = Model_Evaluation()
hist_me.plot_hist_graph(hist)
#%% Model Saving

with open (MMS_FILE_NAME,'wb') as file:
    pickle.dump(mms,file)


#%% Model Deployment(test_data)

CSV_TEST_PATH = os.path.join(os.getcwd(),'Top_Glove_Stock_Price_Test.csv')

column_names = ['Date','Open','High','low','Close','Adj_Close','Volume']
test_df = pd.read_csv(CSV_TEST_PATH, names=column_names)


test_df = mms.transform(np.expand_dims(test_df['Open'].values,axis=-1))
con_test = np.concatenate((df,test_df),axis=0)
con_test = con_test[-(win_size+len(test_df)):] # win size + len of test data
x_test = []
for i in range(win_size,len(con_test)): # win size 60,con_test 80
    x_test.append(con_test[i-win_size:i,0])

x_test = np.array(x_test)

predicted = model.predict(np.expand_dims(x_test,axis=-1))


#%% Actual vs Predicted Visualisation

me = Model_Evaluation()
me.plot_predicted_graph(test_df,predicted,mms)


#%% MAE,MSE, MAPE

print('mae:', mean_absolute_error(test_df, predicted))
print('mse:', mean_squared_error(test_df, predicted))
print('mape:', mean_absolute_percentage_error(test_df, predicted))

test_df_inversed = mms.inverse_transform(test_df)
predicted_inversed = mms.inverse_transform(predicted)

print('mae_inverse:', mean_absolute_error(test_df_inversed, predicted_inversed))
print('mse_inverse:', mean_squared_error(test_df_inversed, predicted_inversed))
print('mape_inverse:', mean_absolute_percentage_error(test_df_inversed, predicted_inversed))

#%% Discussion

# This model is able to predict the trend of TopGlove stock price
# Despite error mae is around 4 % and mape is 4.5%
# Can include a web scarping algorithm to analyse the latest news to improve the model performance









