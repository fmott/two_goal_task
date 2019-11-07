# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 09:45:49 2019

@author: ott
"""
import pandas as pd

def tg_preprocess_parameters(filename): 

    params = pd.read_csv(filename)
    
    # Reshape dataframe 
    params_pivot = pd.pivot_table(params, values='value', index=['subjects'], columns=['parameter','variables'])
    
    # Get new column labels
    new_labels_tmp = params_pivot.columns.values
    new_labels = []
    for i in range(len(new_labels_tmp)):
       new_labels.append(new_labels_tmp[i][0]+'_'+new_labels_tmp[i][1])
    
    # Assign new labels     
    params_pivot.columns = new_labels
    
    return params_pivot
