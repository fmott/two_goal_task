# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:39:25 2018

@author: ott
"""
import numpy as np

def tg_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tg_logit(x):
    return np.log(x/(1-x))
