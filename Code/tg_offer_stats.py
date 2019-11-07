# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:27:40 2019

@author: ott
"""

#%% Load stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tg_set_globalplotting import tg_set_globalplotting

dat = pd.read_csv('../Results/preprocessed_results.csv')
tg_set_globalplotting(style='frontiers')
#%% Offers statistics 
n_trials = np.sum(    (dat['phase'] > 1) & (dat['subject'] == 1) )  
# whole experiment
n_offer = np.zeros(4)
for o in range(4):
    n_offer[o]  = np.sum(        (dat['offer'] == o+1) & (dat['phase'] > 1) & (dat['subject'] == 1)  ) 

# trialwise
n_offer_t = np.zeros((4,15))
for o in range(4):
    for t in range(15):
        n_offer_t[o,t]  = np.sum(        (dat['offer'] == o+1) & (dat['phase'] > 1) & (dat['subject'] == 1)  & (dat['trial'] == t+1))     
    
# miniblockwise
n_offer_b = np.zeros((4,60))
for o in range(4):
    idx_b = 0
    for b in range(20):
        for p in range(3):
            n_offer_b[o,idx_b]  = np.sum(        (dat['offer'] == o+1) & (dat['phase'] == p+2) & (dat['subject'] == 1)  & (dat['block'] == b+1))   
            idx_b+=1

# difficulty-wise
n_offer_j = np.zeros((4,3))
for o in range(4):
    for j in range(2,-1,-1):
        n_offer_j[o,j]  = np.sum(        (dat['offer'] == o+1) & (dat['phase'] > 1) & (dat['subject'] == 1) & ( (dat['start_condition'] == 1 + 2*j)|(dat['start_condition'] == 2 + 2*j))   ) 
 
# difficulty-wise & trialwise  
n_offer_jt = np.zeros((4,3,15))
for o in range(4):
    for j in range(2,-1,-1):
        for t in range(15):
            n_offer_jt[o,j,t]  = np.sum(        (dat['offer'] == o+1) & (dat['phase'] > 1) & (dat['subject'] == 1) & ( (dat['start_condition'] == 1 + 2*j)|(dat['start_condition'] == 2 + 2*j)) & (dat['trial'] == t+1)  ) 

# difficulty-wise & blockwise
n_offer_jb = np.zeros((4,3,20))
for o in range(4):
    for j in range(2,-1,-1):
        idx_b = 0
        idx_z = 0
        for b in range(20):
            for p in range(3):
                if np.sum(        (dat['offer'] == o+1) & (dat['phase'] == p + 2) & (dat['subject'] == 1) & ( (dat['start_condition'] == 1 + 2*j)|(dat['start_condition'] == 2 + 2*j))  & (dat['block'] == b+1) ) > 0: 
                    n_offer_jb[o,j,idx_z]  = np.sum(        (dat['offer'] == o+1) & (dat['phase'] == p + 2) & (dat['subject'] == 1) & ( (dat['start_condition'] == 1 + 2*j)|(dat['start_condition'] == 2 + 2*j))  & (dat['block'] == b+1) )    
                    idx_z+=1
                
                idx_b+=1
   
#%% Draw histograms
#%% n_offer 
titles = ['A','B','Ab','aB'] 
fig,axes = plt.subplots(ncols = 1,nrows = 1, figsize = (3.3,2.3))

axes.bar(range(4), n_offer)

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False) 
axes.xaxis.set_tick_params(top='off', direction='out', width=1)
axes.yaxis.set_tick_params(right='off', direction='out', width=1)

axes.set_ylim(0,250)
axes.set_yticks(range(0,250,25))
axes.set_xticks(range(4))
axes.set_xticklabels(titles )
axes.set_ylabel('Number')
axes.set_xlabel('Offer')     

plt.tight_layout()
#fig.savefig('n_offer.png', dpi=300, bbox_inches='tight', transparent=True)
 
#%% n_offer_t                
titles = ['A','B','Ab','aB'] 
fig,ax = plt.subplots(ncols = 2,nrows = 2, figsize = (7,4))

for i, axes in enumerate(ax.flat):
    axes.bar(range(15), n_offer_t[i,:])
    axes.set_title(titles[i])

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False) 
    axes.xaxis.set_tick_params(top='off', direction='out', width=1)
    axes.yaxis.set_tick_params(right='off', direction='out', width=1)
    
    axes.set_ylim(0,25)
    axes.set_xticks(range(0,15,2))
    axes.set_xticklabels(range(1,16,2) )
    axes.set_ylabel('Number')
    axes.set_xlabel('Trial')

plt.tight_layout()
#fig.savefig('n_offer_t.png', dpi=300, bbox_inches='tight', transparent=True)
  
#%% n_offer_b
titles = ['A','B','Ab','aB'] 
k=1
fig,ax = plt.subplots(ncols = 2,nrows = 2, figsize = (7*k,4*k))

for i, axes in enumerate(ax.flat):
    axes.bar(range(60), n_offer_b[i,:],width = 0.6)
    axes.set_title(titles[i])

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False) 
    axes.xaxis.set_tick_params(top='off', direction='out', width=1)
    axes.yaxis.set_tick_params(right='off', direction='out', width=1)
    
    axes.set_ylim(0,8)
    axes.set_xlim(0,61)
    axes.set_xticks(range(4,64,5))
    axes.set_xticklabels(range(5,65,5) )
    axes.set_ylabel('Number')
    axes.set_xlabel('Miniblock')

plt.tight_layout()
#fig.savefig('n_offer_b.png', dpi=300, bbox_inches='tight', transparent=True)
    

#%% n_offer_j
titles = ['A','B','Ab','aB'] 
conds = ['easy', 'medium', 'hard']
k=1
fig,ax = plt.subplots(ncols = 3,nrows = 1, figsize = (7*k,1.8*k))

for i, axes in enumerate(ax.flat):
    axes.bar(range(4), n_offer_j[:,i])
    axes.set_title(conds[i])

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False) 
    axes.xaxis.set_tick_params(top='off', direction='out', width=1)
    axes.yaxis.set_tick_params(right='off', direction='out', width=1)
    
    axes.set_ylim(0,85)
    axes.set_xticks(range(4))
    axes.set_xticklabels(titles)
    axes.set_ylabel('Number')
    axes.set_xlabel('Offer')

plt.tight_layout()
#fig.savefig('n_offer_j.png', dpi=300, bbox_inches='tight', transparent=True)
 
#%%  n_offer_jt
titles = ['A','B','Ab','aB'] 
colors =['red', 'green', 'blue']
fig,ax = plt.subplots(ncols = 4,nrows = 3, figsize = (7,4))
idx = 0
for i, axes in enumerate(ax.flat):
    if i < 4:
        axes.bar(range(15), n_offer_jt[i,0,:])
        axes.set_title(titles[i])
    elif (i >=4) & (i < 8) :
        axes.bar(range(15), n_offer_jt[i-4,1,:])
    else:
        axes.bar(range(15), n_offer_jt[i-8,2,:])
        axes.set_xlabel('Trial')

    if i == 0:
        axes.set_ylabel('easy\nNumber')
    elif i == 4:
        axes.set_ylabel('medium\nNumber')
    elif i == 8:
        axes.set_ylabel('hard\nNumber')


    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False) 
    axes.xaxis.set_tick_params(top='off', direction='out', width=1)
    axes.yaxis.set_tick_params(right='off', direction='out', width=1)
    
    axes.set_ylim(0,12)
    axes.set_xticks(range(0,15,2))
    axes.set_xticklabels(range(1,16,2) )

plt.tight_layout()
#fig.savefig('n_offer_jt.png', dpi=300, bbox_inches='tight', transparent=True)

#%%  n_offer_jb
titles = ['A','B','Ab','aB'] 
colors =['red', 'green', 'blue']
fig,ax = plt.subplots(ncols = 4,nrows = 3, figsize = (7,4))
idx = 0
for i, axes in enumerate(ax.flat):
    if i < 4:
        axes.bar(range(20), n_offer_jb[i,0,:])
        axes.set_title(titles[i])
    elif (i >=4) & (i < 8) :
        axes.bar(range(20), n_offer_jb[i-4,1,:])
    else:
        axes.bar(range(20), n_offer_jb[i-8,2,:])
        axes.set_xlabel('Miniblock')

    if i == 0:
        axes.set_ylabel('easy\nNumber')
    elif i == 4:
        axes.set_ylabel('medium\nNumber')
    elif i == 8:
        axes.set_ylabel('hard\nNumber')

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False) 
    axes.xaxis.set_tick_params(top='off', direction='out', width=1)
    axes.yaxis.set_tick_params(right='off', direction='out', width=1)
    
    axes.set_ylim(0,12)
    axes.set_xticks(range(0,20,2))
    axes.set_xticklabels(range(1,21,2) )

plt.tight_layout()
#fig.savefig('n_offer_jb.png', dpi=300, bbox_inches='tight', transparent=True)   

