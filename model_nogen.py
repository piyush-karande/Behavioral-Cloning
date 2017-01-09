# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:03:32 2017

@author: Piyush Karande
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Dropout, Flatten, Input, Activation

import pickle
import numpy as np

from sklearn.model_selection import train_test_split

def get_model():
    # Creat model

    model = Sequential()
    model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, subsample= (2,2), border_mode='valid',input_shape=(64,160,3)))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
    
    model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, subsample= (2,2), border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
    
    model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, subsample= (2,2), border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
    
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
    
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(Dropout(.5))
    
    model.add(Flatten())
    model.add(Dense(100, name='hidden1'))
    model.add(Activation('tanh'))
    model.add(Dropout(.5))
    
    model.add(Dense(50, name='hidden2'))
    model.add(Activation('tanh'))
    model.add(Dropout(.5))
    
    model.add(Dense(10, name='hidden3'))
    model.add(Activation('tanh'))
    model.add(Dropout(.5))
    
    model.add(Dense(1, name='output'))
    model.add(Activation('tanh'))
    
    model.compile(optimizer='adam', loss='mse')
    
    return model


if __name__ == '__main__':
    #Loading data

    train_file = 'track1_train2.pkl'
    recov_file = 'track1_recovery.pkl'
    #train_file2 = 'track2_train.pkl'
    with open(train_file, mode='rb') as f:
        track1_train = pickle.load(f)
        
    with open(recov_file, mode='rb') as f:
        track1_recov = pickle.load(f)
    '''
    with open(train_file2, mode='rb') as f:
        track2_train = pickle.load(f)
    '''
    X_train, y_train = track1_train['train_data'], track1_train['train_angles']
    X_recov, y_recov = track1_recov['train_data'], track1_recov['train_angles']
    #X_train2, y_train2 = track2_train['train_data'], track2_train['train_angles']
    
    del track1_train, track1_recov
    
    X_train = np.append(X_train, X_recov, axis=0)
    #X_train = np.append(X_train, X_train2, axis=0)
    
    y_train = np.append(y_train, y_recov, axis=0)
    #y_train = np.append(y_train, y_train2, axis=0)


    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                        test_size = 0.1,
                                                        random_state = 42)
    
    model = get_model()
    
    history = model.fit(X_train, y_train, 
                        batch_size=100, 
                        nb_epoch=5, 
                        validation_data = (X_test, y_test))
    
    model_json = model.to_json()
    
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
        
    