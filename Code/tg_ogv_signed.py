# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:34:41 2019

@author: ott
"""
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def tg_ogv_signed(dat, subjects=list(range(1,89+1)),trials=list(range(1,15+1)),size='2col',plotting = True, save=False):
    ns = len(subjects)
    nt = len(trials)
    
    helper_idx = -1
    ogv_time = np.zeros((nt,4,ns))
    ogv_time_plus = np.zeros((nt,4,ns))
    ogv_time_minus = np.zeros((nt,4,ns))

    for j in range(2,-2,-1):
        helper_idx+=1
        s_idx = -1
        for s in subjects:
            s_idx += 1
            t_idx = -1
            for t in trials:
                t_idx+=1
                
                if helper_idx < 3: 
                    idx = (dat['trial'] == t)   &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1)  
                    idx_plus = (dat['trial'] == t)   &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1)  & (dat['goal_drive'] > 0)   
                    idx_minus = (dat['trial'] == t)   &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1)  & (dat['goal_drive'] < 0) 
                    ogv_time_plus[t_idx,helper_idx,s_idx] = np.nanmean(dat.loc[idx_plus,['goal_drive']].values)
                    ogv_time_minus[t_idx,helper_idx,s_idx] = np.nanmean(dat.loc[idx_minus,['goal_drive']].values)
                    ogv_time[t_idx,helper_idx,s_idx] = np.nanmean(np.absolute(dat.loc[idx,['goal_drive']].values))

                else:
                    idx = (dat['trial'] == t)   &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1)  
                    idx_plus = (dat['trial'] == t)   &  (dat['phase'] > 1)   &  (dat['subject'] == s) & (dat['valid'] == 1) & (dat['goal_drive'] > 0) 
                    idx_minus = (dat['trial'] == t)   &  (dat['phase'] > 1)   &  (dat['subject'] == s) & (dat['valid'] == 1) & (dat['goal_drive'] < 0) 
                    ogv_time_plus[t_idx,helper_idx,s_idx] = np.nanmean(dat.loc[idx_plus,['goal_drive']].values)
                    ogv_time_minus[t_idx,helper_idx,s_idx] = np.nanmean(dat.loc[idx_minus,['goal_drive']].values)
                    ogv_time[t_idx,helper_idx,s_idx] = np.nanmean(np.absolute(dat.loc[idx,['goal_drive']].values))

        
    mean_ogv_time_plus = np.nanmean(ogv_time_plus,2)
    std_ogv_time_plus = np.nanstd(ogv_time_plus,2)  
    mean_ogv_time_minus = np.nanmean(ogv_time_minus,2)
    std_ogv_time_minus = np.nanstd(ogv_time_minus,2)   
    mean_ogv_time = np.nanmean(ogv_time,2)
    std_ogv_time = np.nanstd(ogv_time,2)   



    if plotting == True:
        if size == '1col':
          width  = 3.3  
          height = 2.3
        elif size == '2col':
          width  = 7
          height = 2.3
        elif size == 'big':
          width  = 7*3
          height = 7.5*3
    
        tick_length = 2
        tick_width = 1
        lw=1

        xticks1 = range(0,15,2)
        xticklabels1 = range(1,16,2)
        subplot_labels =['A','B','C','D']
        cols = ['red','green','blue']
        red_patch = mpatches.Patch(color='red', label='easy')
        green_patch = mpatches.Patch(color='green', label='medium')
        blue_patch = mpatches.Patch(color='blue', label='hard')
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(width,height))
        plt.tight_layout()

        for i in range(3): 
            ax[0].plot(mean_ogv_time[:,i],color=cols[i],linewidth=lw,linestyle='-')
            ax[0].fill_between(range(15),mean_ogv_time[:,i]+std_ogv_time[:,i], mean_ogv_time[:,i]-std_ogv_time[:,i], facecolor=cols[i], alpha=0.2)
            ax[0].plot([-0.2,14.2],[0,0],linewidth=0.5,color = 'black')

            ax[1].plot(mean_ogv_time_plus[:,i],color=cols[i],linewidth=lw)
            ax[1].fill_between(range(15),mean_ogv_time_plus[:,i]+std_ogv_time_plus[:,i], mean_ogv_time_plus[:,i]-std_ogv_time_plus[:,i], facecolor=cols[i], alpha=0.2)
            ax[1].plot(mean_ogv_time_minus[:,i],color=cols[i],linewidth=lw,linestyle='--')
            ax[1].fill_between(range(15),mean_ogv_time_minus[:,i]+std_ogv_time_minus[:,i], mean_ogv_time_minus[:,i]-std_ogv_time_minus[:,i], facecolor=cols[i], alpha=0.2)
            ax[1].plot([-0.2,14.2],[0,0],linewidth=0.5,color = 'black')
            
           
        for i,axes in enumerate(ax.flat):   
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False) 
            axes.xaxis.set_tick_params(top='off', direction='out', width=1)
            axes.yaxis.set_tick_params(right='off', direction='out', width=1)
            axes.tick_params(length=tick_length, width=tick_width)
            axes.set_xlabel('Trials')
            axes.set_xticks(xticks1)
            axes.set_xticklabels(xticklabels1)
            axes.set_yticks(range(-5,6))
            axes.set_yticklabels(range(-5,6))
            axes.text(-0.15,1.2,subplot_labels[i],transform=axes.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')

            if i == 0: 
                axes.set_ylim(-3,3)
                axes.set_ylabel('Differential expected value (DEV)')
                axes.legend(handles=[red_patch,green_patch,blue_patch],loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3,frameon=False)

            else:
                axes.set_ylim(-5,5)
                
            axes.set_xlim(-0.2,14.2)

        if save == True:
            fig.savefig('ogv_cond_signed.png', dpi=300, bbox_inches='tight', transparent=True)
    