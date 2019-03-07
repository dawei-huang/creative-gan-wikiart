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

IMGHEIGHT = 227
IMGWIDTH = 227
CHANNELS = 3

def Preprocess(wdir):
    images, labels = LoadDataTest(wdir)
    imgs = np.zeros((len(images), IMGHEIGHT, IMGWIDTH, CHANNELS))
    for i in range(len(images)):
        images[i] = cv2.resize(images[i],(IMGHEIGHT, IMGWIDTH), interpolation = cv2.INTER_AREA)
        imgs[i, :, :, :] = images[i]
    return imgs, labels        
    
def GenTestSet(images, labels):
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
    images = []
    labels = []
    class_num = -1
    for directory in os.walk(wdir):
        counter = 0
        for img in glob(os.path.join(directory[0], '*.jpg')):
            images.append(cv2.imread(img))
            one_hot = np.zeros(27)
            one_hot[class_num] = 1
            labels.append(one_hot)
            counter += 1
            if counter == 50: #number of images from each class
                break
        class_num += 1
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels


if __name__ == '__main__':
    Preprocess()