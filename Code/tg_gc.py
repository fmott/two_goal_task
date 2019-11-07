# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:52:28 2019

@author: ott
"""
import numpy as np
import pandas as pd
# style = average: return average gc for selected trials and subjects
# style = time: return trialwise average gc for selected subjects and all trials
# subjects: list of subjects to be evaluated
# trials: list of trials to be evaluated

def tg_gc(dat,agdat= pd.DataFrame([]), subjects=list(range(1,89+1)),trials=list(range(1,15+1)),size='1col', style='time', save=False):
    ns = len(subjects)
    nt = len(trials)
    
    if (style == 'average') | (style == 'time'):
        #% Goal choice average
        helper_idx = -1
        p_g2 = np.zeros((4,ns))
        for j in range(2,-2,-1):
            helper_idx+=1
            s_idx = -1
            for s in subjects:
                s_idx += 1
                
                if helper_idx < 3: 
                    idx = (dat['trial'].isin(trials))   &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1) &  ~np.isnan(dat['goal_decision']) 
                    idx_g1 = (dat['trial'].isin(trials))   &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1)   & (dat['goal_decision'] == 1)
                    idx_g2 = (dat['trial'].isin(trials))   &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) &  (dat['valid'] == 1)  & (dat['goal_decision'] == 2)
                    
                    n_elements = np.nansum(idx)
                    n_g1 = np.nansum(idx_g1)
                    n_g2 = np.nansum(idx_g2)
                    p_g2[helper_idx,s_idx] = n_g2/n_elements
                
                else:
                    idx = (dat['trial'].isin(trials))   &  (dat['phase'] > 1)   &  (dat['subject'] == s) & (dat['valid'] == 1) &  ~np.isnan(dat['goal_decision']) 
                    idx_g1 = (dat['trial'].isin(trials))   &  (dat['phase'] > 1)   &  (dat['subject'] == s) & (dat['valid'] == 1) &  (dat['goal_decision'] == 1)
                    idx_g2 = (dat['trial'].isin(trials))   &  (dat['phase'] > 1)   &  (dat['subject'] == s) &  (dat['valid'] == 1) &  (dat['goal_decision'] == 2)
                    
                    n_elements = np.nansum(idx)
                    n_g1 = np.nansum(idx_g1)
                    n_g2 = np.nansum(idx_g2)
                    p_g2[helper_idx,s_idx] = n_g2/n_elements
                
        mean_pg2 = np.nanmean(p_g2,1)
        std_pg2 = np.nanstd(p_g2,1)
                
        
        if  (style == 'time'):
        #% Goal choice trialwise
            helper_idx = -1
            p_g2_time = np.zeros((nt,4,ns))
            for j in range(2,-2,-1):
                helper_idx+=1
                s_idx = -1
                for s in subjects:
                    s_idx += 1
                    t_idx = -1
                    for t in trials:
                        t_idx+=1
                        
                        if helper_idx < 3: 
                            idx = (dat['trial'] == t)   &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1) &  ~np.isnan(dat['goal_decision']) 
                            idx_g1 = (dat['trial'] == t)   &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1)   & (dat['goal_decision'] == 1)
                            idx_g2 = (dat['trial'] == t)    &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) &  (dat['valid'] == 1)  & (dat['goal_decision'] == 2)
                            
                            n_elements = np.nansum(idx)
                            n_g1 = np.nansum(idx_g1)
                            n_g2 = np.nansum(idx_g2)
                            p_g2_time[t_idx,helper_idx,s_idx] = n_g2/n_elements
                        
                        else:
                            idx = (dat['trial'] == t)   &  (dat['phase'] > 1)   &  (dat['subject'] == s) & (dat['valid'] == 1) &  ~np.isnan(dat['goal_decision']) 
                            idx_g1 = (dat['trial'] == t)   &  (dat['phase'] > 1)   &  (dat['subject'] == s) & (dat['valid'] == 1) &  (dat['goal_decision'] == 1)
                            idx_g2 = (dat['trial'] == t)   &  (dat['phase'] > 1)   &  (dat['subject'] == s) &  (dat['valid'] == 1) &  (dat['goal_decision'] == 2)
                            
                            n_elements = np.nansum(idx)
                            n_g1 = np.nansum(idx_g1)
                            n_g2 = np.nansum(idx_g2)
                            p_g2_time[t_idx,helper_idx,s_idx] = n_g2/n_elements
                
            mean_pg2_time = np.nanmean(p_g2_time,2)
            std_pg2_time = np.nanstd(p_g2_time,2)   
    
## Agent
    if not agdat.empty:             
        if (style == 'average') | (style == 'time'):
                #% Goal choice average
                helper_idx = -1
                p_g2_agent = np.zeros((4,ns))
                for j in range(2,-2,-1):
                    helper_idx+=1
                    s_idx = -1
                    for s in subjects:
                        s_idx += 1
                        
                        if helper_idx < 3: 
                            idx = (agdat['trial'].isin(trials))   &  (agdat['phase'] > 1)  & ( (agdat['start_condition'] == 1+2*j) |  (agdat['start_condition'] == 2+2*j) )  &  (agdat['subject'] == s) & (agdat['valid'] == 1) &  ~np.isnan(agdat['goal_decision']) 
                            idx_g1 = (agdat['trial'].isin(trials))   &  (agdat['phase'] > 1)  & ( (agdat['start_condition'] == 1+2*j) |  (agdat['start_condition'] == 2+2*j) )  &  (agdat['subject'] == s) & (agdat['valid'] == 1)   & (agdat['goal_decision'] == 1)
                            idx_g2 = (agdat['trial'].isin(trials))   &  (agdat['phase'] > 1)  & ( (agdat['start_condition'] == 1+2*j) |  (agdat['start_condition'] == 2+2*j) )  &  (agdat['subject'] == s) &  (agdat['valid'] == 1)  & (agdat['goal_decision'] == 2)
                            
                            n_elements = np.nansum(idx)
                            n_g1 = np.nansum(idx_g1)
                            n_g2 = np.nansum(idx_g2)
                            p_g2_agent[helper_idx,s_idx] = n_g2/n_elements
                        
                        else:
                            idx = (agdat['trial'].isin(trials))   &  (agdat['phase'] > 1)   &  (agdat['subject'] == s) & (agdat['valid'] == 1) &  ~np.isnan(agdat['goal_decision']) 
                            idx_g1 = (agdat['trial'].isin(trials))   &  (agdat['phase'] > 1)   &  (agdat['subject'] == s) & (agdat['valid'] == 1) &  (agdat['goal_decision'] == 1)
                            idx_g2 = (agdat['trial'].isin(trials))   &  (agdat['phase'] > 1)   &  (agdat['subject'] == s) &  (agdat['valid'] == 1) &  (agdat['goal_decision'] == 2)
                            
                            n_elements = np.nansum(idx)
                            n_g1 = np.nansum(idx_g1)
                            n_g2 = np.nansum(idx_g2)
                            p_g2_agent[helper_idx,s_idx] = n_g2/n_elements
                        
                mean_pg2_agent = np.nanmean(p_g2_agent,1)
                std_pg2_agent = np.nanstd(p_g2_agent,1)
                        
                
                if  (style == 'time'):
                #% Goal choice trialwise
                    helper_idx = -1
                    p_g2_time_agent = np.zeros((nt,4,ns))
                    for j in range(2,-2,-1):
                        helper_idx+=1
                        s_idx = -1
                        for s in subjects:
                            s_idx += 1
                            t_idx = -1
                            for t in trials:
                                t_idx+=1
                                
                                if helper_idx < 3: 
                                    idx = (agdat['trial'] == t)   &  (agdat['phase'] > 1)  & ( (agdat['start_condition'] == 1+2*j) |  (agdat['start_condition'] == 2+2*j) )  &  (agdat['subject'] == s) & (agdat['valid'] == 1) &  ~np.isnan(agdat['goal_decision']) 
                                    idx_g1 = (agdat['trial'] == t)   &  (agdat['phase'] > 1)  & ( (agdat['start_condition'] == 1+2*j) |  (agdat['start_condition'] == 2+2*j) )  &  (agdat['subject'] == s) & (agdat['valid'] == 1)   & (agdat['goal_decision'] == 1)
                                    idx_g2 = (agdat['trial'] == t)    &  (agdat['phase'] > 1)  & ( (agdat['start_condition'] == 1+2*j) |  (agdat['start_condition'] == 2+2*j) )  &  (agdat['subject'] == s) &  (agdat['valid'] == 1)  & (agdat['goal_decision'] == 2)
                                    
                                    n_elements = np.nansum(idx)
                                    n_g1 = np.nansum(idx_g1)
                                    n_g2 = np.nansum(idx_g2)
                                    p_g2_time_agent[t_idx,helper_idx,s_idx] = n_g2/n_elements
                                
                                else:
                                    idx = (agdat['trial'] == t)   &  (agdat['phase'] > 1)   &  (agdat['subject'] == s) & (agdat['valid'] == 1) &  ~np.isnan(agdat['goal_decision']) 
                                    idx_g1 = (agdat['trial'] == t)   &  (agdat['phase'] > 1)   &  (agdat['subject'] == s) & (agdat['valid'] == 1) &  (agdat['goal_decision'] == 1)
                                    idx_g2 = (agdat['trial'] == t)   &  (agdat['phase'] > 1)   &  (agdat['subject'] == s) &  (agdat['valid'] == 1) &  (agdat['goal_decision'] == 2)
                                    
                                    n_elements = np.nansum(idx)
                                    n_g1 = np.nansum(idx_g1)
                                    n_g2 = np.nansum(idx_g2)
                                    p_g2_time_agent[t_idx,helper_idx,s_idx] = n_g2/n_elements
                        
                    mean_pg2_time_agent = np.nanmean(p_g2_time_agent,2)
                    std_pg2_time_agent = np.nanstd(p_g2_time_agent,2)
                    
                    # Difference
                    mean_diff_pg2 = mean_pg2 - mean_pg2_agent
                    std_diff_pg2 = std_pg2 + std_pg2_agent
                    mean_diff_pg2_time = mean_pg2_time - mean_pg2_time_agent
                    std_diff_pg2_time = std_pg2_time + std_pg2_time_agent

    #%% Return formatted 
    if style == 'time':
        return mean_pg2, std_pg2, p_g2, mean_pg2_time,std_pg2_time
    elif style == 'average':
        return mean_pg2, std_pg2, p_g2
    
