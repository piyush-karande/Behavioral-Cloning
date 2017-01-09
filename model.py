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
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def train_generator(driving_log, recovery_log, batch_size=100, reco_ratio = .25):
   
    while 1:
        
        reco_size = int(batch_size*reco_ratio)
        reco_ind = np.random.randint(len(recovery_log), size=reco_size)
        reco_imgs, reco_angles = load_images(recovery_log.ix[reco_ind])
        
        train_size = batch_size-reco_size
        train_ind = np.random.randint(len(driving_log), size=train_size)
        train_imgs, train_angles = load_images(driving_log.ix[train_ind])
        
        train_imgs = np.append(train_imgs, reco_imgs, axis=0)
        train_angles = np.append(train_angles, reco_angles, axis=0)
        
        yield train_imgs, train_angles

def load_images(df, cam_flag=1, v_crop=16):
    imgs = np.zeros([len(df),64,160,3])
    angles = np.zeros(len(df))
    
    for ind in range(len(df)):
        if cam_flag:
            cam = np.random.randint(3)
        else:
            cam = 0

        img_file = df.iloc[ind][cam]
        img = plt.imread(img_file)
        
        angle = df.iloc[ind]['SteeringAngle']
        
        if np.random.randint(2):
            img = cv2.flip(img,1)
            angle = -1*angle
        
        img = cv2.resize(img, (160,80),interpolation = cv2.INTER_AREA)
        img = img[v_crop:,:,:]
        img = img/255.0 - .5
        imgs[ind] = img
        
        
        if cam==1: angle += 0.15
        if cam==2: angle -= 0.15
        
        angles[ind] = angle

    angles[angles>1.] = 1.
    angles[angles<-1.] = -1.
    
    return imgs, angles

def get_model():
    # Creat model

    model = Sequential()
    model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, subsample= (2,2), border_mode='valid',input_shape=(64,160,3)))
    model.add(Activation('elu'))
    #model.add(Dropout(.5))
    
    model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, subsample= (2,2), border_mode='valid'))
    model.add(Activation('elu'))
    #model.add(Dropout(.5))
    
    model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, subsample= (2,2), border_mode='valid'))
    model.add(Activation('elu'))
    #model.add(Dropout(.5))
    
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid'))
    model.add(Activation('elu'))
    model.add(Dropout(.5))
    
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid'))
    model.add(Activation('elu'))
    model.add(Dropout(.5))
    
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

    col_names = ['Center', 'Left', 'Right', 'SteeringAngle', 'Throttle', 
                 'Break', 'Speed']
                 
    driving_log_file = r'C:\Users\Piyush Karande\Desktop\simulator-windows-64\Track 1\Train\driving_log.csv'
    
    recovery_log_file = r'C:\Users\Piyush Karande\Desktop\simulator-windows-64\Track 1\Recovery\driving_log.csv'
    
    driving_log = pd.read_csv(driving_log_file, names=col_names, 
                              skipinitialspace=True)
    
    recovery_log = pd.read_csv(recovery_log_file, names=col_names, 
                              skipinitialspace=True)
    
    temp = np.random.randint(len(driving_log), size=len(driving_log)//10)
    
    validation_log = driving_log.ix[temp]
    
    #driving_log = driving_log.drop(temp)
    
    #imgs, angles = load_images(validation_log.iloc[:100])
    
    
    model = get_model()
    
    img_generator = train_generator(driving_log, recovery_log)
    
    
    history = model.fit_generator(img_generator,
                                  samples_per_epoch=20000,
                                  nb_epoch=5)
    
    model_json = model.to_json()
    
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
        
