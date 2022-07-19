# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:42:25 2022

@author: pc
"""


import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Input
import numpy as np


class EDA():
    def __init__(self):
        pass
    
    
    def plot_graph(self,df):
        '''
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.plot(df['Open'])
        plt.plot(df['High'])
        plt.plot(df['Low'])
        plt.legend(['Open','High','Low'])
        plt.show()

class ModelCreation():
    def __init__(self):
        pass
    
    def simple_lstm_layer(self,x_train,num_node=128,drop_rate=0.3,
                          output_node=1):
        model = Sequential()
        model.add(Input(shape=(np.shape(x_train)[1],1))) # Input 
        model.add(LSTM(num_node,return_sequences=(True)))
        model.add(Dropout(drop_rate))
        model.add(LSTM(num_node))
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node,activation='linear'))
        model.summary()
        
        return model


class Model_Evaluation():
    def plot_predicted_graph(self,test_df,predicted,mms):
        plt.figure()
        plt.plot(test_df,'b',label='actual stock price')
        plt.plot(predicted,'r',label='predicted stock price')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(mms.inverse_transform(test_df),'b',label='actual stock price')
        plt.plot(mms.inverse_transform(predicted),'r',label='predicted stock price')
        plt.legend()
        plt.show()
        
    def plot_hist_graph(self,hist):
        plt.figure()
        plt.plot(hist.history['mape'])
        plt.show()

        plt.figure()
        plt.plot(hist.history['loss'])
        plt.show()
    
    
    
    
    
    