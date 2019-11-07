# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:27:10 2019

@author: ott
"""
import pandas as pd
import numpy as np

dat = pd.read_csv('../Results/preprocessed_results.csv')

# Offers & Conditions
offer_effect = np.array([[1,0],[0,1],[1,-1],[-1,1]])
idx = (dat['phase'] > 1) & (dat['subject'] == 1) 
offers = dat.loc[idx, ['offer']]
offers_arr = np.reshape(offers.to_numpy(), (60,15))

conditions = dat.loc[idx, ['start_condition']]
conditions_arr = np.reshape(conditions.to_numpy(), (60,15))
condition_effect = np.array([[7,5],[5,7],[8,6],[6,8],[9,7],[7,9]])

threshold = 11

# Generate all possible trajectories
n_trajectories = 2**15
trajectories = np.zeros((n_trajectories, 15))
for i in range(n_trajectories): 
    trajectories[i,:]  = list(map(int,str(    np.binary_repr(i,width=15)    )))
    
# Get performance    
# - Get offer effect sequence given an offer sequence in a miniblock and multiply it by all possible trajectories  
# - Sum up the points add initial points and evalauate goal success
success = np.zeros(60)  
for b in range(60):
    gain_per_trial = np.tile( offer_effect[offers_arr[b,:]-1 ].T, (len(trajectories),1,1))   * trajectories[:,np.newaxis,:] 
    gain_per_miniblock = np.sum(gain_per_trial,2)
    start_points = condition_effect[conditions_arr[b,0]-1]
    points_per_miniblock = gain_per_miniblock + start_points
    
    if np.sum ((points_per_miniblock[:,0] >= threshold) & (points_per_miniblock[:,1] >= threshold) ) > 1:
        success[b] = 2
    elif  np.sum ( (points_per_miniblock[:,0] >= threshold)  ^  (points_per_miniblock[:,1] >= threshold)  ) > 1:
        success[b] = 1
    else:
        success[b] = 0
    
# G2 success per condition

easy = (conditions_arr[:,0] == 5) | (conditions_arr[:,0] == 6) 
G2_easy = success[easy] == 2
G1_easy = success[easy] == 1
fail_easy = success[easy] == 0
p_G2_easy = np.sum(G2_easy) / np.sum(easy)
p_G1_easy = np.sum(G1_easy) / np.sum(easy)
p_fail_easy = np.sum(fail_easy) / np.sum(easy)

medium = (conditions_arr[:,0] == 3) | (conditions_arr[:,0] == 4) 
G2_medium = success[medium] == 2
G1_medium = success[medium] == 1
fail_medium = success[medium] == 0
p_G2_medium = np.sum(G2_medium) / np.sum(medium)
p_G1_medium = np.sum(G1_medium) / np.sum(medium)
p_fail_medium = np.sum(fail_medium) / np.sum(medium)

hard = (conditions_arr[:,0] == 1) | (conditions_arr[:,0] == 2) 
G2_hard = success[hard] == 2
G1_hard = success[hard] == 1
fail_hard = success[hard] == 0
p_G2_hard = np.sum(G2_hard) / np.sum(hard)
p_G1_hard = np.sum(G1_hard) / np.sum(hard)
p_fail_hard = np.sum(fail_hard) / np.sum(hard)






    