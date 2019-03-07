# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 06:25:24 2019

@author: daniel
"""
import Preprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

def ZFNet():
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=7, strides=2, padding='same', activation='relu', input_shape=(Preprocess.IMGHEIGHT, Preprocess.IMGWIDTH,  Preprocess.CHANNELS)))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    
    model.add(ZeroPadding2D(padding=2))
    model.add(Conv2D(filters=256, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    
    model.add(Conv2D(filters=512, padding='same', kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=1024, padding='same', kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=512, padding='same', kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    model.add(Dense(4096), activation='relu')
    model.add(Dropout(0.5))
    model.add(Dense(1000), activation='relu')
    model.add(Dense(200), activation='relu')
    model.add(Dense(27), activation='softmax')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def ConvNet():
    model = Sequential()
    #add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(Preprocess.IMGHEIGHT, Preprocess.IMGWIDTH,  Preprocess.CHANNELS)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(27, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    ZFNet()
    ConvNet()