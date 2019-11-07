# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:50:32 2019

@author: ott
"""
import numpy as np
import matplotlib.pyplot as plt

def tg_plot_selfreport_vs_params(dat,params,save=False): 
    #%% Reported strategy vs strategy preference 
    reported_strat = dat.loc[  (dat['trial']==1)  &  (dat['block']==1)   & (dat['phase']==1) ]      ['reported_strategy'].values
    strat_pref = params['bias_2_median'].values
    
    pref_sq =  strat_pref[reported_strat == 1]
    pref_par = strat_pref[reported_strat == 2]
    pref_mix = strat_pref[reported_strat == 3]
        
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(3.3,2.2))
    plt.tight_layout()
    ax.plot(reported_strat + ( (np.random.rand(len(reported_strat))-0.5)/8),strat_pref,'o',color='orange',markeredgecolor='black',markersize=4,alpha = 0.4)
    ax.boxplot([pref_sq,pref_par,pref_mix], 0, '')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)       
    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', direction='out', width=1)
    ax.set_ylabel('Strategy preference ' + r'($\theta$)')
    ax.set_xlabel('Reported strategy')
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(['Sequential','Parallel','Mixed'])
    
    if save == True:
        fig.savefig('reported_strat_vs_strat_pref.png', dpi=300, bbox_inches='tight', transparent=True)