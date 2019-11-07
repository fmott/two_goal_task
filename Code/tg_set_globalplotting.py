# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:05:15 2019

@author: ott
"""
import matplotlib.pyplot as plt

def tg_set_globalplotting(style= 'frontiers'):
    if style == 'frontiers':
        plt.rcParams['axes.labelsize'] = 9
        plt.rcParams['axes.titlesize'] = 9
        plt.rcParams['axes.linewidth'] = 0.6
        plt.rcParams['xtick.labelsize'] = 7
        plt.rcParams['ytick.labelsize'] = 7
        plt.rcParams['legend.fontsize'] = 8
        plt.rcParams["font.family"] = 'serif'
