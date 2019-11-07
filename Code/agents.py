#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:50:01 2018

@author: Dimitrije Markovic
modified by Florian Ott 
"""

import torch
import numpy as np
from torch.distributions import Categorical

ones = torch.ones
zeros = torch.zeros

def get_sign_switcher(ns, L, no):
    # define sign switcher (negative or positive multiplier of the bias) 
    # for specific state offer pairs 
    sign_switcher = ones(ns, no)
    
    map_states_to_points = torch.tensor(
            np.array(np.unravel_index(np.arange(ns), (L, L))).T
            )
    
    pointdiff = map_states_to_points[:, 0] - map_states_to_points[:, 1]
    
    negative = pointdiff < 0
    positive = pointdiff > 0
    equal_zero = pointdiff == 0
    
    sign_switcher[negative, 3] = -1. 
    sign_switcher[positive, 2] = -1. 
    sign_switcher[equal_zero, 2:] = -1 
    
    return sign_switcher

class Random(object):
    #this agent generates random responses
    def __init__(self, runs = 1, trials = 1, na = 2):
        
        self.na = na #number of actions
        
        self.cat = Categorical(probs = ones(runs, self.na))
        
    def update_beliefs(self, block, trial, *args):
        pass
        
    def planning(self, block, trial, *args):
        pass
            
    def sample_responses(self, trial):
        
        return self.cat.sample()

class Informed(object):
    def __init__(self, 
                 offer_state_tm,
                 state_outcomes,
                 offer_probability,
                 runs = 1,
                 mini_blocks = 1,
                 trials = 15, 
                 na = 2,
                 ns = 16**2,
                 no = 4,
                 lam = 5.):
        
        self.runs = runs
        self.nb = mini_blocks
        self.nt = trials
        
        self.na = na #number of actions
        self.ns = ns #number of states
        self.L = np.sqrt(ns).astype(int) #max number of points
        self.no = no

        self.offer_state_tm = offer_state_tm
        self.offer_probability = offer_probability.reshape(-1, 1, 1)
        self.lam = lam
        self.state_outcomes = state_outcomes
        
        self.sign_switcher = get_sign_switcher(self.ns, self.L, self.no)
        self.map_offer_to_type = torch.tensor([0, 0, 1, 1], dtype = torch.long)
        
        self.range = torch.tensor(range(self.runs))
        
    def set_parameters(self, x=None):
        if x is not None:
            self.beta = x[:,:2].exp() #decision noise for two offer types
            self.bias = x[:,2:4] #choice bias for two offer typpes
            self.gamma = x[:,4].sigmoid() # future value discount rate
            self.kappa = 2*x[:,5].sigmoid() 
            if x.shape[-1] > 6:
                time = torch.arange(self.nt, dtype=torch.float).reshape(1, -1)
                self.dynamic = True
                
                dynbeta = x[:, 1].reshape(-1, 1) + x[:, 6].reshape(-1, 1) * time/self.nt
                        
                dynbias = x[:, 3].reshape(-1, 1) + x[:, 7].reshape(-1, 1) * time/self.nt
                
                self.dynbeta = zeros(self.runs, 2, self.nt)
                self.dynbeta[:, 0] = self.beta[:, 0].reshape(-1, 1)
                self.dynbeta[:, 1] = dynbeta.exp()
                
                self.dynbias = zeros(self.runs, 2, self.nt)
                self.dynbias[:, 0] = self.bias[:, 0].reshape(-1, 1)
                self.dynbias[:, 1] = dynbias
                
                self.a = x[:, 6]
                self.b = x[:, 7]
                self.npars = 8
            else:
                self.dynamic = False
                self.npars = 6
        else:
            self.npars = 6
            self.dynamic = False
            self.beta = 1e10*ones(self.runs, 2)
            self.bias = zeros(self.runs, 2)
            self.gamma = ones(self.runs)
            self.kappa = ones(self.runs)
            
        #State values
        self.V = [] #zeros(self.nt, self.ns, self.runs)
        self.Q = [] #zeros(self.nt, self.na, self.no, self.ns, self.runs)
        
        subjective_state_outcomes = self.state_outcomes.reshape(-1 ,1).repeat(1, self.runs)
        locs = subjective_state_outcomes == 2*self.lam
        subjective_state_outcomes[locs] = \
            (self.kappa * subjective_state_outcomes[locs].reshape(-1, self.runs)).reshape(-1)
    
        #self.V[0] = subjective_state_outcomes
        self.V.append(subjective_state_outcomes)
        
        self.logprobs = zeros(self.runs, self.nb, self.nt, self.na)
        
        self.compute_state_values()
        
        return self
        
    def compute_state_values(self):
        tm = self.offer_state_tm
        op = self.offer_probability
        gamma = self.gamma
        
        for d in range(1,self.nt+1):
            Q = tm.matmul(self.V[d-1])            
            
            vmax, _ = Q.max(dim=0)
            
            self.Q.append(Q)
            if d < self.nt:
                self.V.append(gamma*torch.sum(vmax*op, dim=0))
                
    def update_beliefs(self, block, trial, states, offers):
        self.current_state = states
        self.current_offer = offers
        
    def planning(self, block, trial):
        states = self.current_state
        offers = self.current_offer
     
        d = self.nt - trial - 1 #remaining trials
        
        Q = self.Q[d]
        
        offer_type = self.map_offer_to_type.index_select(0, offers)
        sign = self.sign_switcher[states, offers]
        
        if self.dynamic:
            betas = self.dynbeta[self.range, offer_type, trial]
            biases = self.dynbias[self.range, offer_type, trial]
        else:
            betas = self.beta[self.range, offer_type]
            biases = self.bias[self.range, offer_type]
            
        self.logprobs[:, block, trial, 0] = Q[0, offers, states, self.range]*betas 
        self.logprobs[:, block, trial, 1] = Q[1, offers, states, self.range]*betas + biases*sign
            
    def sample_responses(self, block,trial):
        cat = Categorical(logits=self.logprobs[:, block, trial])
        return cat.sample()