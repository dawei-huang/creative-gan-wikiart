# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 04:29:50 2019

@author: daniel
"""

import cv2
import os
from glob import glob
import numpy as np
#import matplotlib.pyplot as plt

IMGHEIGHT = 32
IMGWIDTH = 32
CHANNELS = 3

def Preprocess(wdir):
    images, labels = LoadDataTest(wdir)
    #Do Preprocessing here
    return images, labels        
    
def GenTestSet(images, labels):
    #Split data into Training and Testing Set with 80/20 split
    p = int(0.2 * images.shape[0])
    idx = np.random.permutation(images.shape[0])
    train_idx, test_idx = idx[p:], idx[:p]
    train_img, test_img = images[train_idx], images[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    return train_img, train_labels, test_img, test_labels

def LoadData(wdir):
    images = np.asarray([cv2.imread(img) for directory in os.walk(wdir) for img in glob(os.path.join(directory[0], '*.png'))])
    labels = np.asarray([directory[0] for directory in os.walk(wdir) for img in glob(os.path.join(directory[0], '*.png'))])
    return images, labels

def LoadDataTest(wdir):
    #Count number of files and Preallocate Arrays
    num = sum([len(files) for r, d, files in os.walk(wdir)])
    images = np.empty((num, IMGHEIGHT, IMGWIDTH, CHANNELS))
    labels = np.empty((num, 27))
    #Iterate through all images
    class_num = -1
    counter = -1
    for directory in os.walk(wdir):
        num_img = 0
        for img in glob(os.path.join(directory[0], '*.jpg')):
            #Resize before adding to array
            images[counter, :, :, :] = cv2.resize(cv2.imread(img),(IMGHEIGHT, IMGWIDTH), interpolation=cv2.INTER_AREA)
            one_hot = np.zeros(27)
            one_hot[class_num] = 1
            labels[counter, :] = one_hot
            counter += 1
            num_img += 1
            if num_img == 10:
                break
        class_num += 1
    return images, labels


if __name__ == '__main__':
    Preprocess()