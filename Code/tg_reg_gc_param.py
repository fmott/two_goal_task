# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:48:01 2019

@author: ott
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from tg_gc import tg_gc

def tg_reg_gc_param(dat,params,save=False):
    #%% Multiple regression GC ~ model paramters with interactions
    gc_mean,gc_sd,gc1 = tg_gc(dat,trials=[1,2,3,4,5,6,7],style='average')
    gc_mean,gc_sd,gc2 = tg_gc(dat,trials=[8,9,10,11,12,13,14,15],style='average')
    gc = np.hstack([gc1[-1,:],gc2[-1,:]])
    first_half = np.hstack( [np.ones((89)) , np.zeros((89)) ] )
    
    y = gc
    X = np.zeros((89*2,6))
    X[:,0] = np.hstack ( [   params['bias_2_median'].values, params['bias_2_median'].values])
    X[:,1] = np.hstack ( [params['beta_2_median'].values, params['beta_2_median'].values])
    X[:,2] = np.hstack ( [params['gamma_median'].values,  params['gamma_median'].values])
    X[:,3] = np.hstack ( [params['kappa_median'].values, params['kappa_median'].values])
    
    X[:,4] = first_half
    X[:,5] = first_half*X[:,0] # interaction strat pref and miniblock
    
    X = sm.add_constant(X)
    lm7 = sm.OLS(y,X).fit()
    print(lm7.summary())
    
    # Extract stuff
    lm7_params = lm7.params
    lm7_se = lm7.bse
    lm7_p_values = lm7.pvalues

    # Predicted Data 
    y_pred_1= lm7.params[0]  +   lm7.params[1] * X[0:89,1]       +   lm7.params[2] * X[0:89,2].mean()   +   lm7.params[3] * X[0:89,3].mean()   +   lm7.params[4] * X[0:89,4].mean() +   lm7.params[5] * 1 +   lm7.params[6] * X[0:89,1] * 1
    y_pred_2 = lm7.params[0]  +   lm7.params[1] * X[89:,1]      +   lm7.params[2] * X[89:,2].mean()   +   lm7.params[3] * X[89:,3].mean()   +   lm7.params[4] * X[89:,4].mean() +   lm7.params[5] * 0 +   lm7.params[6] * X[89:,1] * 0    
    xx = X[0:89,1]
    
    #%% Just plot the above horizontally
    ylabel = 'Coefficient value'
    xticks = np.array([0,1,2,3,4,5])
    xticklabels = ['Strategy Preference','Precision', 'Discount', 'Reward Ratio', 'First half', 'First half X        \n Strategy Preference'] 
    subplot_labels=['A','B','C','D']
    tick_length = 2
    tick_width = 1
    
    fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(7,2.2))
    plt.tight_layout()
    
    # Improve visualization of large errorbar
    ax[0].axvline(x=2,color='black',linewidth=1,linestyle='--',alpha=0.5)
    ax[0].plot([2,2],[lm7_params[3]-0.3,lm7_params[3]+0.08],color='black',linewidth=1,linestyle='-')
    lm7_se[3] = np.nan
    
    ax[0].bar([0,1,2,3,4,5], lm7_params[1:], width=0.5, color = 'grey', yerr=lm7_se[1:] ,error_kw=dict(elinewidth=1 ),alpha=0.8)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)   
    ax[0].spines['bottom'].set_position('zero')
    ax[0].set_ylim(-0.2,0.3)
    #ax[0].set_ylim(-0.02,0.6)
    
    ax[0].xaxis.set_tick_params(top='off', direction='out', width=1)
    ax[0].yaxis.set_tick_params(right='off', direction='out', width=1) 
    ax[0].set_xticks(xticks+0.1)
    ax[0].set_xticklabels(xticklabels,rotation=40, ha='right')
    ax[0].set_ylabel(ylabel)
    ax[0].tick_params(axis='x',length=0, width=tick_width,left=True,bottom=True)
    ax[0].text(-0.1,1.1,subplot_labels[0],transform=ax[0].transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')
    ax[0].tick_params(axis='x', which='major', pad=55)
  
    for i in range(1,7):
        if lm7_params[i] >= 0: 
            if (lm7_p_values[i] < 0.05) & (lm7_p_values[i] > 0.01):
                ax[0].text(i-1,lm7_params[i]+[lm7_se[i]],'*',fontsize=10,fontweight='bold',horizontalalignment = 'center')
           
            elif (lm7_p_values[i] < 0.01) & (lm7_p_values[i] > 0.001):
                ax[0].text(i-1,lm7_params[i]+[lm7_se[i]],'**',fontsize=10,fontweight='bold',horizontalalignment = 'center')
            
            elif (lm7_p_values[i] < 0.001):
                ax[0].text(i-1,lm7_params[i]+[lm7_se[i]],'***',fontsize=10,fontweight='bold',horizontalalignment = 'center')
        else:
            if (lm7_p_values[i] < 0.05) & (lm7_p_values[i] > 0.01):
                ax[0].text(i-1,lm7_params[i]-[lm7_se[i]],'*',fontsize=10,fontweight='bold',horizontalalignment = 'center')
           
            elif (lm7_p_values[i] < 0.01) & (lm7_p_values[i] > 0.001):
                ax[0].text(i-1,lm7_params[i]-[lm7_se[i]],'**',fontsize=10,fontweight='bold',horizontalalignment = 'center')
            
            elif (lm7_p_values[i] < 0.001):
                ax[0].text(i-1,lm7_params[i]-[lm7_se[i]]-0.05,'***',fontsize=10,fontweight='bold',horizontalalignment = 'center')
           
    #Visualize interaction 
    xlabel = ['Strategy Preference'] 
    ylabel = 'Proportion g2-choice'
    xticks = [ [-0.5,0,0.5,1,1.5,2],[1,3,5,7,9,11,13], [0.8, 0.85, 0.9, 0.95, 1], [0.5,1,1.5,2]]
    xtick_labels = [ [-0.5,0,0.5,1,1.5,2], [1,3,5,7,9,11,13],[0.8, 0.85, 0.9, 0.95, 1], [0.5,1,1.5,2] ]
    subplot_labels=['A','B','C','D']
    tick_length = 2
    tick_width = 1
    y_ticks = [0.25,0.5,0.75]
    y_tick_labels = y_ticks
    a_patch = mpatches.Patch(color='black', label='First half')
    b_patch = mpatches.Patch(color='maroon', label='Second half')
    
    ax[1].plot(params['bias_2_median'].values,gc1[-1,:],'o',color='black',markersize=4,alpha = 0.6)
    ax[1].plot(params['bias_2_median'].values,gc2[-1,:],'o',color='maroon',markersize=4,marker="^",alpha = 0.6)
    ax[1].plot(xx,y_pred_1,color='black',)
    ax[1].plot(xx,y_pred_2,color='maroon')
    
    i = 0   
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)       
    ax[1].xaxis.set_tick_params(top='off', direction='out', width=1)
    ax[1].yaxis.set_tick_params(right='off', direction='out', width=1)
    ax[1].set_xlabel(xlabel[i])
    ax[1].set_ylabel(ylabel)
    ax[1].tick_params(length=tick_length, width=tick_width,left=True,bottom=True)
    ax[1].set_xticks(xticks[i])
    ax[1].set_xticklabels(xtick_labels[i])
    ax[1].set_yticks(y_ticks)
    ax[1].set_yticklabels(y_tick_labels)
    ax[1].legend(handles=[a_patch,b_patch],loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2,frameon=False)
    ax[1].text(-0.1,1.1,subplot_labels[1],transform=ax[1].transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=None)

    
    if save == True:
        fig.savefig('lm_gc_params_interaction.png', dpi=300, bbox_inches='tight', transparent=True)
