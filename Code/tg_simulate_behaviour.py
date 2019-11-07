# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:47:53 2018

@author: ott
"""

import torch
import numpy as np

from tasks import TwoGoalTask
from agents import Informed
from simulate import Simulator

from helpers import offer_state_mapping
from tg_preprocessing_sim import tg_preprocessing_sim

def tg_simulate_behaviour(dat,param_sample):

    dat = dat.loc[dat['phase'] > 1]
    
    #%% Define initialization variables 
    runs = len(param_sample) #number of parallel runs of the experiment (e.g. number of subjects)
    mini_blocks = 60 # number of mini-blocks
    trials = 15 #number of trials per mini block
    na = 2 #number of choices in each trial
    no = 4 #number of different offers (A, B, Ab, Ba)
    threshold = 10 #goal threshold
    lam = 5 #value of the one goal
    op = torch.FloatTensor([1/no, 1/no, 1/no, 1/no]) #offer probability
    trial_values = dat.loc[:, 'score_A_before':'score_B_before'].values - 1 # Note that a state (8,6) in python corresponds to a state (9,7) in the csv data file
    L = trial_values.max()+1
    ns = L**2 #number of states
    ns = np.asscalar(ns)
    
    # Get sequence of offers. The sequence of offers was the same for all subjects. 
    offers_tmp = dat['offer'].values.reshape(89, -1, trials)-1
    offers_tmp = offers_tmp[0,:,:]
    offers = torch.tensor(np.repeat(offers_tmp[np.newaxis,:,:],runs,axis=0))
    
    # Get initial states 
    states = torch.tensor(np.ravel_multi_index(trial_values.T, (L, )*2).T.reshape(89, -1, trials))
    initial_states = torch.zeros(runs, mini_blocks, dtype=torch.int32)
    initial_states[:] = states[3, :, 0][None, :]
    
    # Define state transition matrix for the environment p(s'|s, o, a)
    offer_state_tm = torch.zeros(na, no, ns, ns) # note that the order of elements is inverted -> a, o, s| s'
    offer_state_tm[0] = torch.eye(ns).repeat(no, 1, 1) # a = 0 corresponds to wait action, states don't change in this case
    offer_state_tm[1] = offer_state_mapping(L, no) # a = 1 coresponds to accepting an offer
    
    # Set state reward mapping on the end trial R(s)
    outcomes = np.sum(np.indices((L, L)) >= threshold, axis=0).flatten()
    outcomes = outcomes*lam
    outcomes = torch.from_numpy(outcomes).float()
    
    #%% Set up the environment
    env = TwoGoalTask(offer_state_tm, 
                      runs=runs,
                      mini_blocks=mini_blocks,
                      trials=trials,
                      number_states=ns, 
                      number_offers=no, 
                      initial_states=initial_states,
                      offers=offers)
    
    #%% Set up agent. Set agent parameters (bias, precision, discount,...). 
    # prameters log_beta_1, log_beta_2, bias_1, bias_2, logit(gamma)
    agent = Informed(offer_state_tm,
                     outcomes,
                     op,
                     runs=runs,
                     mini_blocks=mini_blocks,
                     trials=trials,
                     na=na,
                     ns=ns,
                     no=no)
    
    #%% Simulate responses
#    trans_parameters = torch.tensor([100000, 100000, 0, 0, 10.,0.]).repeat(runs, 1) # define transformed parameters

    agent = agent.set_parameters(torch.Tensor(param_sample.values))
    sim = Simulator(env, agent, runs=runs, mini_blocks=mini_blocks, trials=trials)
    sim.simulate_experiment()
    
    #%% Preprocessing of simulated data
    df_sim = tg_preprocessing_sim(sim,threshold)
    
    return df_sim
