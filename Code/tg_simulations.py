# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:10:03 2019

@author: Florian Ott
"""

import numpy as np
import pandas as pd
import glob as glob 

from tg_set_globalplotting import tg_set_globalplotting
from tg_simulate_behaviour import tg_simulate_behaviour 
from tg_suboptimal_goal_choice import tg_suboptimal_goal_choice
from tg_suboptimal_goal_choice_sim import tg_suboptimal_goal_choice_sim
from tg_performance_sim import tg_performance_sim

tg_set_globalplotting(style='frontiers')
dat = pd.read_csv('../Results/preprocessed_results.csv')

#%% S6-7 Figure
#Random agent 

n_sample = 1000
param_sample = pd.DataFrame()
param_sample['beta_1'] = np.ones(n_sample)*10
param_sample['beta_2'] = np.ones(n_sample)*-100
param_sample['bias_1'] = np.ones(n_sample)*100
param_sample['bias_2'] = np.ones(n_sample)*0
param_sample['gamma'] = np.ones(n_sample)*10
param_sample['kappa'] = np.ones(n_sample)*0

# Simulate behaviour
df_sim = tg_simulate_behaviour(dat,param_sample)

# Plot simulated suboptimal goal choice
tmp = tg_suboptimal_goal_choice_sim(df_sim, save=False)

# Plot simulated performance 
tg_performance_sim(df_sim,save=False)


#%% S10-11 Figure
# Posterior predictive checks 
#Load data frame
filename = glob.glob('../Results/theta_beta_gamma_kappa/group_posterior_sample*.csv')[0] 
param_sample_group = pd.read_csv(filename)

n_sample = 1000
param_sample = pd.DataFrame()
param_sample['beta_1'] = np.ones(n_sample)*-100
param_sample['beta_2'] = param_sample_group['beta']
param_sample['bias_1'] = np.ones(n_sample)*100
param_sample['bias_2'] = param_sample_group['bias']
param_sample['gamma'] = param_sample_group['gamma']
param_sample['kappa'] = param_sample_group['kappa']

# Simulate behaviour
df_sim = tg_simulate_behaviour(dat,param_sample)

# Plot simulated suboptimal goal choice
tmp = tg_suboptimal_goal_choice(df_sim, save=False)

# Plot simulated performance 
tg_performance_sim(df_sim, save=False)

#%% S1-2 Movie
# Variable beta & performance/suboptimal gc across time
# Load data frame
filename = glob.glob('./results/theta_beta_gamma_kappa/group_posterior_sample*.csv')[0] # theta_beta_gamma_kappa
param_sample_group = pd.read_csv(filename)

n_sample = 1000
param_sample = pd.DataFrame()

for i,value in enumerate(np.arange(0.25,3.25,0.25)):
    param_sample['beta_1'] = np.ones(n_sample)*-100
    param_sample['beta_2'] = np.ones(n_sample)*np.log(value)
    param_sample['bias_1'] = np.ones(n_sample)*100
    param_sample['bias_2'] = param_sample_group['bias']
    param_sample['gamma'] = param_sample_group['gamma']
    param_sample['kappa'] = param_sample_group['kappa']

    # Simulate behaviour
    df_sim = tg_simulate_behaviour(dat,param_sample)

    # Plot simulated suboptimal goal choice
    tmp = tg_suboptimal_goal_choice_sim(df_sim, save=False, beta = True, value = value)
    sub_g_time = {value:tmp}

    # Plot simulated performance 
    tg_performance_sim(df_sim,save=False,beta = True, value = value)

#%% S3-4 Movie
# Variable theta & performance/suboptimal gc across time
# Load and preprocess data frame
filename = glob.glob('./results/theta_beta_gamma_kappa/group_posterior_sample*.csv')[0] # theta_beta_gamma_kappa
param_sample_group = pd.read_csv(filename)

n_sample = 1000
param_sample = pd.DataFrame()

for i,value in enumerate(np.arange(-1,1.25,0.25)):
    param_sample['beta_1'] = np.ones(n_sample)*-100
    param_sample['beta_2'] = param_sample_group['beta']
    param_sample['bias_1'] = np.ones(n_sample)*100
    param_sample['bias_2'] = np.ones(n_sample)*value
    param_sample['gamma'] = param_sample_group['gamma']
    param_sample['kappa'] = param_sample_group['kappa']

    # Simulate behaviour
    df_sim = tg_simulate_behaviour(dat,param_sample)

    # Plot simulated suboptimal goal choice
    tmp = tg_suboptimal_goal_choice_sim(df_sim, save=False,theta = True, value = value)
    res_subg = {value:tmp}
    
    # Plot simulated performance 
    tg_performance_sim(df_sim,save=False,theta = True, value = value)  