# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:32:36 2019

@author: ott
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tg_suboptimal_goal_choice import tg_suboptimal_goal_choice

def tg_plot_subg_hilobias(dat,params,save = False):
    params['bias_2_median'].min()
    
    highest = params['bias_2_median'].idxmax()
    highest_val = params.loc[highest,['bias_2_median']]
    lowest = params['bias_2_median'].idxmin()
    lowest_val = params.loc[lowest,['bias_2_median']]
    
    _,_,_,_,_,subg1,_ = tg_suboptimal_goal_choice(dat,subjects=[lowest],plotting=False)
    _,_,_,_,_,subg2,_ = tg_suboptimal_goal_choice(dat,subjects=[highest],plotting=False)
    
    #%% Plotting 
    
       
    barx1 = [0,1,3,4,6,7,9,10]
    bary1 = subg1
    barerr1 = np.zeros(len(subg1))
    barx2 = [0,1,3,4,6,7,9,10]
    bary2 = subg2
    barerr2 = np.zeros(len(subg2))
    
    ylabel = ['Suboptimal g-choice','Suboptimal g-choice','Suboptimal g2-choice','Suboptimal g1-choice']
    subplot_labels =['A','B','C','D']
    
    xticks2 = barx1
    xticklabels2 = ['g2','g1','g2','g1','g2','g1','g2','g1']
    tick_length = 2
    tick_width = 1
    red_patch = mpatches.Patch(color='red', label='easy')
    green_patch = mpatches.Patch(color='green', label='medium')
    blue_patch = mpatches.Patch(color='blue', label='hard')
    grey_patch = mpatches.Patch(color='grey', label='all')
    colors =  ['red','green','blue','grey']
    titles =['Low Strategy \n Preference','High Strategy \n Preference' ]
    
    
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(3.3,1.4))
    plt.tight_layout()
    
    for i in range(8):
     ax[0].bar(barx1[i], bary1[i], width=0.8, color = colors[(np.floor(i/2).astype(int))], yerr=barerr1[i] ,error_kw=dict(elinewidth=1 ),alpha=0.5)
     ax[1].bar(barx2[i], bary2[i], width=0.8, color = colors[(np.floor(i/2).astype(int))], yerr=barerr2[i] ,error_kw=dict(elinewidth=1 ),alpha=0.5)
     
    for i ,axes in enumerate(ax.flat):   
        axes.text(-0.15,1.2,subplot_labels[i],transform=axes.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')
    
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False) 
        
        axes.xaxis.set_tick_params(top='off', direction='out', width=1)
        axes.yaxis.set_tick_params(right='off', direction='out', width=1)
        axes.set_ylim((0,0.7))
        axes.tick_params(length=tick_length, width=tick_width)
        axes.set_xticks(xticks2)
        axes.set_xticklabels(xticklabels2)
        axes.tick_params(axis='x',labelsize=6)
        axes.set_title(titles[i])
    
        if i == 0:
            axes.legend(handles=[red_patch,green_patch,blue_patch,grey_patch],loc='upper center', bbox_to_anchor=(1.1, 1.7), ncol=4,frameon=False)
            axes.set_ylabel(ylabel[i],fontsize=7) 
    
    if save == True:
        fig.savefig('subg_hilobias.png', dpi=300, bbox_inches='tight', transparent=True)
    
    return highest_val,lowest_val