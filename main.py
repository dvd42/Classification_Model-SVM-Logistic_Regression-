#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:03:17 2017

@author: diego
"""

        #print "Gamma",gamma
        #print "Kernel",kernel
        #print "C",C
        #print "Degree",degree
        #print "Probability",probability


import Classifier as c
import Data_preprocessing as d

num = 5
method = "svm"

x,y = d.load_data()

split = d.kfold(x,num)
# Make SVM parameters as runtime global parameters
print c.validate(x,y,split,method)

