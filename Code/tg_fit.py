# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:52:34 2019

@author: ott
"""
import time as time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents import Informed
from inference import Inferrer
from helpers import offer_state_mapping

zeros = torch.zeros
ones = torch.ones

import pyro
pyro.validation_enabled()
#%% Define submodel and saving location
n_subjects = 89 #number of subjects
#model = 'theta'
#model = 'beta'
#model = 'theta_beta'
#model = 'theta_beta_gamma'
#model = 'theta_beta_kappa'
model = 'theta_beta_gamma_kappa'

if model == 'theta':
    vals = torch.ones(n_subjects, 5)
    vals[:, 0] = -100.
    vals[:, 1] = -100.
    vals[:, 2] = 100.
    vals[:, 3] = 10.
    vals[:, 4] = 0.

    fixed_params = {'labels': [0,1,2,4,5],
                    'values': vals}
    labels2 = ['bias']
    directory = '.\\results_tmp\\theta\\'

elif model == 'beta':
    vals = torch.ones(n_subjects, 5)
    vals[:, 0] = -100.
    vals[:, 1] = 100.
#    vals[:, 2] = 100.
    vals[:, 2] = 0.
    vals[:, 3] = 10.
    vals[:, 4] = 0.

    fixed_params = {'labels': [0,2,3,4,5],
                    'values': vals}
    labels2 = ['beta']
    directory = '.\\results_tmp\\beta\\'
    
elif model == 'theta_beta':
    vals = torch.ones(n_subjects, 4)
    vals[:, 0] = -100.
    vals[:, 1] = 100.
    vals[:, 2] = 10.
    vals[:, 3] = 0.

    fixed_params = {'labels': [0,2,4,5],
                    'values': vals}
    labels2 = ['beta','bias']
    directory = '.\\results_tmp\\theta_beta\\'
    
elif model == 'theta_beta_gamma':
    vals = torch.ones(n_subjects, 3)
    vals[:, 0] = -100.
    vals[:, 1] = 100.
    vals[:, 2] = 0.

    fixed_params = {'labels': [0,2,5],
                    'values': vals}
    labels2 = ['beta','bias','gamma']
    directory = '.\\results_tmp\\theta_beta_gamma\\'
    
elif model == 'theta_beta_kappa':
    vals = torch.ones(n_subjects, 3)
    vals[:, 0] = -100
    vals[:, 1] = 100.
    vals[:, 2] = 10.
    
    fixed_params = {'labels': [0,2,4],
                    'values': vals}
    labels2 = ['beta','bias','kappa']
    directory = '.\\results_tmp\\theta_beta_kappa\\'
    
elif model == 'theta_beta_gamma_kappa':
    vals = torch.ones(n_subjects, 2)
    vals[:, 0] = -100.
    vals[:, 1] = 100.

    fixed_params = {'labels': [0,2],
                    'values': vals}
    labels2 = ['beta','bias','gamma','kappa']
    directory = '.\\results_tmp\\theta_beta_gamma_kappa\\'
    
#%% load and preprocess data
tmp = pd.read_csv('../Results/preprocessed_results.csv')
trials = 15 # number of trials
trial_values = tmp.loc[:, 'score_A_before':'score_B_before'].values - 1
L = trial_values.max()+1 #max number of states along one dimension

states = torch.tensor(np.ravel_multi_index(trial_values.T, (L, )*2).T.reshape(n_subjects, -1, trials))
point_diff = torch.tensor(tmp['score_difference'].values.reshape(n_subjects, -1, trials), dtype=torch.float)
offers = torch.tensor(tmp['offer'].values.reshape(n_subjects, -1, trials)-1)
responses = torch.tensor(2-tmp['response'].values.reshape(n_subjects, -1, trials))#map response labels to 0-> wait and 1 -> accept 

# exlude trials without clear G-choice classification 
exclude = ((point_diff == 1) & (offers == 3)) | ((point_diff == -1) & (offers == 2))
valid = torch.tensor(tmp['valid'].values.reshape(n_subjects, -1, trials), dtype=torch.bool)
valid = valid & ~exclude & (offers > 1)

# remove training mini-blocks (first 10 mini-blocks)
states = states[:, 10:]
offers = offers[:, 10:]
responses = responses[:, 10:]

# mask invalid trials
mask = valid[:, 10:]

na = 2  # number of actions
no = 4  # number of offers
ns = int(L**2)  # number of states 
mini_blocks = responses.shape[1]  # number of mini-blocks
threshold = 10  # goal threshold
offer_prob = torch.FloatTensor([1/no, 1/no, 1/no, 1/no]) #offer probability
lam = 5 #value of reaching one goal

# Transition matrix
offer_state_tm = zeros(na, no, ns, ns)#define state transition matrix for the environment p(s'|s, o, a). #note that the order of elements is inverted -> a, o, s| s'
offer_state_tm[0] = torch.eye(ns).repeat(no, 1, 1)#a = 0 corresponds to wait action, states don't change in this case
offer_state_tm[1] = offer_state_mapping(L, no)#a = 1 coresponds to accepting an offer

#Set state reward mapping on the end trial R(s)
outcomes = np.sum(np.indices((L, L)) >= threshold, axis=0).flatten()
outcomes = outcomes*lam
outcomes = torch.from_numpy(outcomes).float()

# Define model
agent = Informed(offer_state_tm,
                 outcomes,
                 offer_prob,
                 runs=n_subjects,
                 mini_blocks=mini_blocks,
                 trials=trials,
                 ns=ns)
x = zeros(n_subjects, 6)
agent.set_parameters(x)

# Infer paramters 
stimulus = {'states':states, 'offers': offers}
infer = Inferrer(agent, stimulus, responses, mask=mask, fixed_params=fixed_params)
infer.infer_posterior(iter_steps=200, parametrisation='cent-horseshoe')

#plot convergence of stochasitc ELBO estimates (log-model evidence)
plt.plot(infer.loss[-150:])

# Format results
labels = ['beta_1', 'beta_2', 'bias_1', 'bias_2', 'gamma', 'kappa']
posterior_percentiles = infer.format_results(labels, n_samples=1000)

# Sample posterior
subject_posterior, _, group_posterior = infer.sample_from_posterior(labels2, n_samples=1000)[:3]

elbo = pd.DataFrame()     
elbo['elbo'] = infer.loss



#%%  Save results of the parameter fitting
timestr = time.strftime("%Y%m%d-%H%M%S")
name = directory + 'posterior_median_percentiles_'+ timestr + '.csv'
posterior_percentiles.to_csv(name)

# save elbo
name = directory + 'elbo_'+ timestr + '.csv'
elbo.to_csv(name)

# Save posterior sample subjects
name = directory + 'subject_posterior_sample_'+ timestr + '.csv'
subject_posterior.to_csv(name)

# Save posterior sample group
name = directory + 'group_posterior_sample_'+ timestr + '.csv'
group_posterior.to_csv(name)












