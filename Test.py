# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 04:35:38 2019

@author: daniel
"""

import Preprocess
import MyZFNet
#import matplotlib.pyplot as plt

images, labels = Preprocess.Preprocess('..\\wikiart\\')
train_imgs, train_labels, test_imgs, test_labels = Preprocess.GenTestSet(images, labels)

model = MyZFNet.ConvNet()

model.fit(train_imgs, train_labels, validation_data=(test_imgs, test_labels), epochs=5)

