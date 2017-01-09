# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:00:57 2016

@author: Piyush Karande
"""
import numpy as np
import pandas as pd
import cv2
import pickle
import matplotlib.pyplot as plt

def load_imgs(img_files, v_crop=16):
    images = np.zeros([len(img_files),64,160,3])
    
    for ind in range(len(img_files)):
        if not ind%1000:
            print(ind, '/', len(img_files))
        img = plt.imread(img_files[ind])
        img = cv2.resize(img, (160,80),interpolation = cv2.INTER_AREA)
        #img = np.expand_dims(img, 0)
        img = img[v_crop:,:,:]
        img = img/255.0 - .5
        #if len(images)==0:
         #   images = img
        #else:
        images[ind] = img
        
    #images = images/255.0 - .5
        
    return images.astype(np.float16)

def load_driving_log(driving_log_file, col_names, v_crop=16, cam_flag=0):
    
    driving_log = pd.read_csv(driving_log_file, names=col_names, 
                              skipinitialspace=True)
    
    steering_angles = np.array(driving_log['SteeringAngle'])
    center_files = np.array(driving_log['Center'])
    right_files = np.array(driving_log['Right'])
    left_files = np.array(driving_log['Left'])
    
    if cam_flag==0:
        images = load_imgs(center_files, v_crop)
        angles = steering_angles
        
    elif cam_flag==1:
        images = load_imgs(left_files, v_crop)
        angles = steering_angles + 0.2
        
    else:
        images = load_imgs(right_files, v_crop)
        angles = steering_angles - 0.2

    return images, angles

if __name__ == '__main__':
    
    driving_log_file = r'C:\Users\Piyush Karande\Desktop\simulator-windows-64\Track 1\Train\driving_log.csv'
    col_names = ['Center', 'Left', 'Right', 'SteeringAngle', 'Throttle', 
                 'Break', 'Speed']
    center_imgs, center_angles = load_driving_log(driving_log_file, col_names)
    
    
    left_imgs, left_angles = load_driving_log(driving_log_file, 
                                         col_names, 
                                         cam_flag=1)
    
    right_imgs, right_angles = load_driving_log(driving_log_file, 
                                                col_names, 
                                                cam_flag=2)
    
    train_imgs = np.append(center_imgs,left_imgs,axis=0)
    train_imgs = np.append(train_imgs,right_imgs,axis=0)
    
    train_angles = np.append(center_angles,left_angles,axis=0)
    train_angles = np.append(train_angles,right_angles,axis=0)
    
    train_angles[train_angles>1.]=1.
    train_angles[train_angles<-1.]=-1.
    
    data = {'train_data': train_imgs,
            'train_angles': train_angles}
    
    del right_imgs, left_imgs, center_imgs, train_imgs
    
    pickle_file = open('track1_train2_new.pkl', 'wb')
    pickle.dump(data, pickle_file)
    pickle_file.close()
    