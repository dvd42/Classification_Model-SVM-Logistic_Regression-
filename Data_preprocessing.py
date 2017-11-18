#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:30:06 2017

@author: diego
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold


def load_data():
    dataset = pd.read_csv("Wholesale customers data.csv")    
    x = dataset.iloc[:,2:].values
    y = dataset.iloc[:,0].values
    #partitions = [0.5, 0.7, 0.8]

    sc = MinMaxScaler()

    x = sc.fit_transform(x)
    
    return x,y


def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    n_train = int(np.floor(x.shape[0] * train_ratio))
    indices_train = indices[:n_train]
    indices_test = indices[n_train:]

    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_test = x[indices_test, :]
    y_test = y[indices_test]

    return x_train, y_train, x_test, y_test



# Apply K-Fold Cross Validation Split
def kfold(x,num):
    kf = KFold(n_splits=num,shuffle=True)
    return kf.split(x)

""" 
#Apply Random Split
def split(x,y,partitions,method="logistic"):
    
    score = 0.0    
    for part in partitions:
        
        x_t,y_t,x_v,y_v = split_data(x,y,part)
        score = classify(x_t, y_t, x_v, y_v,method="svm",C=1,kernel='rbf',degree=3,gamma=1,probability=False)
        
        print "Ratio: %.1f Score: %f" % (part,score)        

"""
