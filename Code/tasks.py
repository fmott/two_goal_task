#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  22 14:50:01 2018

@author: Dimitrije Markovic
"""

import torch
from torch.distributions import Categorical

ones = torch.ones
zeros = torch.zeros

class TwoGoalTask(object):
    def __init__(self, 
                 transition_matrix, 
                 runs=1,
                 mini_blocks=1,
                 trials=15,
                 number_states=15**2,
                 number_offers=4,
                 initial_states=None,
                 offers_probability=None,
                 offers=None):
        
        self.ns = number_states
        self.no = number_offers
        
        self.tm = transition_matrix
        self.states = zeros(runs, mini_blocks, trials+1, dtype = torch.long)
        if initial_states is None:
            cat = Categorical(probs = ones(runs, mini_blocks, self.ns))
            self.states[:,:,0] = cat.sample()
        else:
            self.states[:,:,0] = initial_states
        
        self.offers = zeros(runs, mini_blocks, trials, dtype=torch.long)
        if offers_probability is None:
            if offers is None:
                self.cat_offers = Categorical(probs = ones(runs,self.no))
            else:
                self.offers = offers
                self.cat_offers = None
        else:
            self.cat_offers = Categorical(probs = offers_probability.repeat(runs,1))
            
        if self.cat_offers is not None:
            self.offers[:,0,0] = self.cat_offers.sample()
    
    def update_environment(self, block, trial, responses):
        
        offers = self.offers[:,block,trial-1]
        states = self.states[:,block,trial-1]
        
        cat_states = Categorical(probs=self.tm[responses, offers, states])
        
        self.states[:,block,trial] = cat_states.sample()
        
        if trial < self.offers.shape[-1] and self.cat_offers is not None:
            self.offers[:,block,trial] =  self.cat_offers.sample()
        