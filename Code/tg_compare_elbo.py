# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:30:57 2019

@author: ott
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tg_set_globalplotting import tg_set_globalplotting

dat = pd.read_csv('../Results/preprocessed_results.csv')
tg_set_globalplotting(style='frontiers')

#%% Elbo 
filename0 = glob.glob('../Results/elbo/theta/elbo*.csv')[0] # theta
filename1 = glob.glob('../Results/elbo/beta/elbo*.csv')[0] # beta
filename2 = glob.glob('../Results/elbo/theta_beta/elbo*.csv')[0] # theta_beta
filename3 = glob.glob('../Results/elbo/theta_beta_kappa/elbo*.csv')[0] # theta_beta_kappa
filename4 = glob.glob('../Results/elbo/theta_beta_gamma_kappa/elbo*.csv')[0] # theta_beta_gamma_kappa

filename5 = glob.glob('../Results/elbo/split_seg_theta/elbo*.csv')[0] # theta
filename6 = glob.glob('../Results/elbo/split_seg_beta/elbo*.csv')[0] # beta
filename7 = glob.glob('../Results/elbo/split_seg_theta_beta/elbo*.csv')[0] # theta_beta
filename8 = glob.glob('../Results/elbo/split_seg_theta_beta_kappa/elbo*.csv')[0] # theta_beta_kappa
filename9 = glob.glob('../Results/elbo/split_seg_theta_beta_gamma_kappa/elbo*.csv')[0] # theta_beta_gamma_kappa

filename10 = glob.glob('../Results/elbo/split_cond_theta/elbo*.csv')[0] # theta
filename11 = glob.glob('../Results/elbo/split_cond_beta/elbo*.csv')[0] # beta
filename12 = glob.glob('../Results/elbo/split_cond_theta_beta/elbo*.csv')[0] # theta_beta
filename13 = glob.glob('../Results/elbo/split_cond_theta_beta_kappa/elbo*.csv')[0] # theta_beta_kappa
filename14 = glob.glob('../Results/elbo/split_cond_theta_beta_gamma_kappa/elbo*.csv')[0] # theta_beta_gamma_kappa

filename15 = glob.glob('../Results/elbo/split_phase_theta_beta_kappa/elbo*.csv')[0] # theta_beta_gamma_kappa



filenames = [filename0,filename1,filename2,filename3,filename4,
             filename5,filename6,filename7,filename8,filename9,
             filename10,filename11,filename12,filename13,filename14,
             filename15]

elbo = np.zeros(len(filenames))
n_iteration = 200
mean_window = 20
elbo_indices = np.arange(n_iteration-1,n_iteration*3*3+n_iteration-1,n_iteration)
elbo_indices = np.array([elbo_indices-mean_window,elbo_indices])
elbo_indices2 = np.arange(n_iteration-1,n_iteration*3+n_iteration-1,n_iteration)
elbo_indices2 = np.array([elbo_indices2-mean_window,elbo_indices2])


for i, filename in enumerate(filenames):
    tmp = pd.read_csv(filename)
    
    if np.isin('condition',tmp.columns) & np.isin('segment',tmp.columns):
        elbo_tmp = 0
        for j in range(9):            
            elbo_tmp = elbo_tmp + np.mean( tmp.loc[elbo_indices[0,j]:elbo_indices[1,j],['elbo']].to_numpy())
        elbo[i] = elbo_tmp
        
    elif ~np.isin('condition',tmp.columns) & np.isin('segment',tmp.columns):
        elbo_tmp = 0
        for j in range(3):            
            elbo_tmp = elbo_tmp + np.mean( tmp.loc[elbo_indices2[0,j]:elbo_indices2[1,j],['elbo']].to_numpy())
        elbo[i] = elbo_tmp

    elif np.isin('condition',tmp.columns) & ~np.isin('segment',tmp.columns):
        elbo_tmp = 0
        for j in range(3):            
            elbo_tmp = elbo_tmp + np.mean( tmp.loc[elbo_indices2[0,j]:elbo_indices2[1,j],['elbo']].to_numpy())
        elbo[i] = elbo_tmp

    else:  
        if np.isin('phase',tmp.columns):
                  elbo_tmp = 0
                  for j in range(3):            
                      elbo_tmp = elbo_tmp + np.mean( tmp.loc[elbo_indices2[0,j]:elbo_indices2[1,j],['elbo']].to_numpy())
                  elbo[i] = elbo_tmp
        else:   
            elbo[i] = np.mean(tmp.loc[len(tmp)-1-mean_window:len(tmp)-1,['elbo']].to_numpy())



# plot elbo
fig,ax = plt.subplots(figsize=(7,3))

labels = [r'$\theta$',r'$\beta$', r'$\theta$'+ r'$\beta$', r'$\theta$'+ r'$\beta$'+ r'$\kappa$', r'$\theta$'+ r'$\beta$'+ r'$\gamma$'+ r'$\kappa$',
          's_'+r'$\theta$','s_'+r'$\beta$','s_'+ r'$\theta$'+ r'$\beta$',  's_'+r'$\theta$'+ r'$\beta$'+ r'$\kappa$','s_'+ r'$\theta$'+ r'$\beta$'+ r'$\gamma$'+ r'$\kappa$',
           'c_'+r'$\theta$','c_'+r'$\beta$', 'c_'+ r'$\theta$'+ r'$\beta$', 'c_'+r'$\theta$'+ r'$\beta$'+ r'$\kappa$','c_'+ r'$\theta$'+ r'$\beta$'+ r'$\gamma$'+ r'$\kappa$',
           'b_'+ r'$\theta$'+ r'$\beta$'+ r'$\kappa$',]

plt.bar(range(len(filenames)),elbo)
plt.xticks(ticks= range(len(filenames)), labels=labels,rotation='vertical')
plt.ylim([19000,22500])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
ax.set_ylabel('Elbo')
ax.set_yticks([19000,19500,20000,20500,21000,21500,22000,22500])
ax.set_yticklabels(-np.array([19000,19500,20000,20500,21000,21500,22000,22500]))

locs,_ = plt.xticks()
ax = plt.gca()
for l in range(len(filenames)):
    plt.text(locs[np.argsort(elbo)[l]],19200, str(l+1),color = 'white', horizontalalignment='center')

#plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
print('Ranked Elbo:', np.argsort(elbo))

