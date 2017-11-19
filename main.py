#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:03:17 2017

@author: diego
"""

import Classifier as c
import Data_preprocessing as d
import runtime_parser as rp
import file_writer as fw

rp.process_runtime_arguments()
x,y,tags = d.load_data(rp.data)

num = x.shape[0] if rp.cv[1] == "n" else float(rp.cv[1])
path = fw.create_dir(rp.cv[0],num,rp.classifier)

if rp.classifier == 1:
    fw.add_file_header(path)

if rp.cv[0] == "kf":
    split = d.kfold(x,int(num))
    c.kf_metrics(x, y, split, path,int(num),tags)

else:
    x_train, x_test, y_train, y_test = d.holdout(x,y,num)
    c.h_metrics(x_train, x_test, y_train, y_test, path,tags)






