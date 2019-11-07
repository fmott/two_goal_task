# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:52:49 2018

@author: ott
"""
import numpy as np
import pandas as pd
import torch

def tg_preprocessing_sim(sim, threshold):
    L = int(np.sqrt(sim.env.ns))
    nr = sim.runs
    nt = sim.nt
    nb = sim.nb
    response_tmp = sim.responses
    unraveled_states_tmp = np.array(np.unravel_index(sim.env.states,(L,L)))
    raveled_states_tmp =  sim.env.states
    score_A_before_tmp = unraveled_states_tmp[0,:,:,:]
    score_B_before_tmp =  unraveled_states_tmp[1,:,:,:]
    offer_tmp = sim.env.offers
    taos_values = torch.stack(sim.agent.Q).numpy()
    ocv_tmp =  taos_values[:,1,:,:,:] - taos_values[:,0,:,:,:]
    
    response = np.zeros(nr*nb*nt,dtype=int)
    score_A_before = np.zeros(nr*nb*nt,dtype=int)
    score_B_before = np.zeros(nr*nb*nt,dtype=int)
    offer = np.zeros(nr*nb*nt,dtype=int)
    subject =  np.zeros(nr*nb*nt,dtype=int)
    block = np.zeros(nr*nb*nt,dtype=int)
    trial = np.zeros(nr*nb*nt,dtype=int)
    valid = np.ones(nr*nb*nt,dtype=int)
    start_condition = np.zeros(nr*nb*nt,dtype=int)
    phase = np.zeros(nr*nb*nt,dtype=int)
    score_difference = np.zeros(nr*nb*nt,dtype=int) 
    ocv = np.zeros(nr*nb*nt)
    ogv = np.zeros(nr*nb*nt)
    goal_decision = np.zeros(nr*nb*nt,dtype=float)
    suboptimal_goal_decision = np.zeros(nr*nb*nt,dtype=float)
    raveled_state = np.zeros(nr*nb*nt,dtype=int)
    idx = 0
    for s in range(nr):
        for b in range(nb):
            for t in range(nt):
                response[idx] = response_tmp[s,b,t]
                score_A_before[idx] = score_A_before_tmp[s,b,t]
                score_B_before[idx] = score_B_before_tmp[s,b,t]
                score_difference[idx] = score_A_before[idx] - score_B_before[idx]
                if ((score_A_before[idx] >= threshold) & (score_B_before[idx] >= threshold)):
                    valid[idx] = 0 
                else:
                    valid[idx] = 1 
    
                if t == 0 :
                    if   ((score_A_before[idx] == 6) & (score_B_before[idx] == 4)):
                        start_condition[idx] = 1    
                    elif ((score_A_before[idx] == 4) & (score_B_before[idx] == 6)):
                        start_condition[idx] = 2   
                    
                    elif ((score_A_before[idx] == 7) & (score_B_before[idx] == 5)):
                        start_condition[idx] = 3   
                    elif ((score_A_before[idx] == 5) & (score_B_before[idx] == 7)):
                        start_condition[idx] = 4   
                    
                    elif ((score_A_before[idx] == 8) & (score_B_before[idx] == 6)):
                        start_condition[idx] = 5   
                    elif ((score_A_before[idx] == 6) & (score_B_before[idx] == 8)):
                        start_condition[idx] = 6
                else:
                    start_condition[idx] = start_condition[idx-1]
    
    
                offer[idx] =  offer_tmp[s,b,t]
                subject[idx] = s+1
                phase[idx] = 99
                block[idx] = b+1
                trial[idx] = t+1
                
                raveled_state[idx] = raveled_states_tmp[s,b,t]
                ocv[idx] = ocv_tmp[15-t-1,offer[idx].astype(int),raveled_state[idx].astype(int),s]
                
                # OGV
                # Score difference > 1  and Score difference < 1
                if (ocv[idx] >= 0)  &  (score_difference[idx] > 1) & (offer[idx] == 2):
                    ogv[idx] = ocv[idx]*-1
                elif (ocv[idx] >= 0)  &  (score_difference[idx] < -1) & (offer[idx] == 2):
                    ogv[idx] = ocv[idx]
                elif (ocv[idx] >= 0)  &  (score_difference[idx] > 1) & (offer[idx] == 3):
                    ogv[idx] = ocv[idx]
                elif (ocv[idx] >= 0)  &  (score_difference[idx] < -1) & (offer[idx] == 3):
                    ogv[idx] = ocv[idx]*-1
                
                elif (ocv[idx] < 0)  &  (score_difference[idx] > 1) & (offer[idx] == 2):
                    ogv[idx] = ocv[idx]*-1
                elif (ocv[idx] < 0)  &  (score_difference[idx] < -1) & (offer[idx] == 2):
                    ogv[idx] = ocv[idx]
                elif (ocv[idx] < 0)  &  (score_difference[idx] > 1) & (offer[idx] == 3):
                    ogv[idx] = ocv[idx]
                elif (ocv[idx] < 0)  &  (score_difference[idx] < -1) & (offer[idx] == 3):
                    ogv[idx] = ocv[idx]*-1
                    
                # Score difference == 1 and score difference == -1
                elif (ocv[idx] >= 0)  &  (score_difference[idx] == 1) & (offer[idx] == 2):
                    ogv[idx] = ocv[idx]*-1
                elif (ocv[idx] >= 0)  &  (score_difference[idx] == -1) & (offer[idx] == 2):
                    ogv[idx] = np.nan
                elif (ocv[idx] >= 0)  &  (score_difference[idx] == 1) & (offer[idx] == 3):
                    ogv[idx] = np.nan
                elif (ocv[idx] >= 0)  &  (score_difference[idx] == -1) & (offer[idx] == 3):
                    ogv[idx] = ocv[idx]*-1
                
                elif (ocv[idx] < 0)  &  (score_difference[idx] == 1) & (offer[idx] == 2):
                    ogv[idx] = ocv[idx]*-1 
                elif (ocv[idx] < 0)  &  (score_difference[idx] == -1) & (offer[idx] == 2):
                    ogv[idx] = np.nan 
                elif (ocv[idx] < 0)  &  (score_difference[idx] == 1) & (offer[idx] == 3):
                    ogv[idx] = np.nan
                elif (ocv[idx] < 0)  &  (score_difference[idx] == -1) & (offer[idx] == 3):
                    ogv[idx] = ocv[idx]*-1 
                    
                # Score difference == 0
                elif (ocv[idx] >= 0)  &  (score_difference[idx] == 0) & (offer[idx] == 2):
                    ogv[idx] = ocv[idx]*-1
                elif (ocv[idx] >= 0)  &  (score_difference[idx] == 0) & (offer[idx] == 3):
                    ogv[idx] = ocv[idx]*-1
                elif (ocv[idx] < 0)  &  (score_difference[idx] == 0) & (offer[idx] == 2):
                    ogv[idx] = ocv[idx]*-1 
                elif (ocv[idx] < 0)  &  (score_difference[idx] == 0) & (offer[idx] == 3):
                    ogv[idx] = ocv[idx]*-1
                  
                elif  (offer[idx] == 0) | (offer[idx] == 1):
                    ogv[idx] = np.nan
                
                
                
                # Goal choice
                # Score difference > 1  and Score difference < 1
                if (response[idx] == 1)  &  (score_difference[idx] > 1) & (offer[idx] == 2):
                    goal_decision[idx] = 1
                elif (response[idx] == 1)  &  (score_difference[idx] < -1) & (offer[idx] == 2):
                    goal_decision[idx] = 2
                elif (response[idx] == 1)  &  (score_difference[idx] > 1) & (offer[idx] == 3):
                    goal_decision[idx] = 2
                elif (response[idx] == 1)  &  (score_difference[idx] < -1) & (offer[idx] == 3):
                    goal_decision[idx] = 1
                
                elif (response[idx] == 0)  &  (score_difference[idx] > 1) & (offer[idx] == 2):
                    goal_decision[idx] = 2
                elif (response[idx] == 0)  &  (score_difference[idx] < -1) & (offer[idx] == 2):
                    goal_decision[idx] = 1
                elif (response[idx] == 0)  &  (score_difference[idx] > 1) & (offer[idx] == 3):
                    goal_decision[idx] = 1
                elif (response[idx] == 0)  &  (score_difference[idx] < -1) & (offer[idx] == 3):
                    goal_decision[idx] = 2
                    
                # Score difference == 1 and score difference == -1
                elif (response[idx] == 1)  &  (score_difference[idx] == 1) & (offer[idx] == 2):
                    goal_decision[idx] = 1
                elif (response[idx] == 1)  &  (score_difference[idx] == -1) & (offer[idx] == 2):
                    goal_decision[idx] = np.nan
                elif (response[idx] == 1)  &  (score_difference[idx] == 1) & (offer[idx] == 3):
                    goal_decision[idx] = np.nan
                elif (response[idx] == 1)  &  (score_difference[idx] == -1) & (offer[idx] == 3):
                    goal_decision[idx] = 1
                
                elif (response[idx] == 0)  &  (score_difference[idx] == 1) & (offer[idx] == 2):
                    goal_decision[idx] = 2 
                elif (response[idx] == 0)  &  (score_difference[idx] == -1) & (offer[idx] == 2):
                    goal_decision[idx] = np.nan 
                elif (response[idx] == 0)  &  (score_difference[idx] == 1) & (offer[idx] == 3):
                    goal_decision[idx] = np.nan
                elif (response[idx] == 0)  &  (score_difference[idx] == -1) & (offer[idx] == 3):
                    goal_decision[idx] = 2 
                    
                # Score difference == 0
                elif (response[idx] == 1)  &  (score_difference[idx] == 0) & (offer[idx] == 2):
                    goal_decision[idx] = 1
                elif (response[idx] == 1)  &  (score_difference[idx] == 0) & (offer[idx] == 3):
                    goal_decision[idx] = 1
                elif (response[idx] == 0)  &  (score_difference[idx] == 0) & (offer[idx] == 2):
                    goal_decision[idx] = 2 
                elif (response[idx] == 0)  &  (score_difference[idx] == 0) & (offer[idx] == 3):
                    goal_decision[idx] = 2
                
                elif  (offer[idx] == 0) | (offer[idx] == 1):
                    goal_decision[idx] = np.nan 
                    
                
                if (ogv[idx] > 0) & (goal_decision[idx] == 2):
                    suboptimal_goal_decision[idx] = 2 
                elif (ogv[idx] > 0) & (goal_decision[idx] == 1):
                    suboptimal_goal_decision[idx] = -2 
                elif (ogv[idx] < 0) & (goal_decision[idx] == 2):
                    suboptimal_goal_decision[idx] = -1 
                elif (ogv[idx] < 0) & (goal_decision[idx] == 1):
                    suboptimal_goal_decision[idx] = 1 
                else:
                    suboptimal_goal_decision[idx] = np.nan 
                
                idx += 1
    
    #Adapt variables such that they obey the same nomenclature than the behavioural data
    score_A_before = score_A_before+1 
    score_B_before = score_B_before+1
    action_effect = np.array([[1,0],[0,1],[1,-1],[-1,1]]) 
    score_A_after = score_A_before  + action_effect[(offer),0]*response
    score_B_after = score_B_before  + action_effect[(offer),1]*response
    response[response == 0] = 2
    offer = offer+1
    
    df_sim = pd.DataFrame( {'response' : response,
                   'score_A_before' : score_A_before,
                   'score_B_before' : score_B_before,
                   'score_A_after' : score_A_after,
                   'score_B_after' : score_B_after,
                   'offer' : offer,
                   'start_condition' : start_condition,
                   'trial' : trial,
                   'block' : block,
                   'phase' : phase,
                   'subject' : subject,
                   'valid' : valid,
                   'score_difference' : score_difference,
                   'dv' : ocv,
                   'goal_drive' : ogv,
                   'goal_decision' : goal_decision,
                   'suboptimal_goal_decision': suboptimal_goal_decision,
                   'raveled_state': raveled_state})
        
    return df_sim