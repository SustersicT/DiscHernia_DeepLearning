# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 11:43:09 2021

@author: Tijana
"""
import keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense, Dropout

def discnet():
    model = Sequential()
    
    model.add(Conv2D(filters=16, kernel_size=(7, 7), activation='relu', input_shape=(64,64,3)))
    model.add(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))

    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dropout(0.4)) 
    model.add(BatchNormalization())
    model.add(Dense(units=3, activation = 'softmax'))
    
    return model