# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:09:01 2019

@author: ott
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def tg_suboptimal_goal_choice(dat, subjects=list(range(1,89+1)),trials=list(range(1,15+1)), size='2col',save=False,plotting = True):
    ns = len(subjects)
    nt = len(trials)
    #%% Average
    
    helper_idx = -1
    p_subg2 = np.zeros((4,ns))
    p_subg1 = np.zeros((4,ns))
    p_subg =  np.zeros((4,ns))
    for j in range(2,-2,-1):
        helper_idx+=1
        s_idx = -1
        for s in subjects:
            s_idx += 1
            if helper_idx < 3: 
                idx = (dat['trial'].isin(trials))   &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1) &  ~np.isnan(dat['suboptimal_goal_decision']) 
                idx_subg1 = (dat['trial'].isin(trials))  &  (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1) &    (dat['suboptimal_goal_decision'] == -2)
                idx_subg2 = (dat['trial'].isin(trials))  &  ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) &  (dat['valid'] == 1) &   (dat['suboptimal_goal_decision'] == -1)
                
                n_elements = np.nansum(idx)
                n_subg1 = np.nansum(idx_subg1)
                n_subg2 = np.nansum(idx_subg2)
                p_subg2[helper_idx,s_idx] = n_subg2/n_elements
                p_subg1[helper_idx,s_idx] = n_subg1/n_elements
                p_subg[helper_idx,s_idx]  = (n_subg2+n_subg1)/n_elements

        
            else:
                idx =  (dat['trial'].isin(trials))  &  (dat['phase'] > 1)   &  (dat['subject'] == s) & (dat['valid'] == 1) &  ~np.isnan(dat['suboptimal_goal_decision']) 
                idx_subg1 =  (dat['trial'].isin(trials))  &  (dat['phase'] > 1)  &  (dat['subject'] == s) & (dat['valid'] == 1) &    (dat['suboptimal_goal_decision'] == -2)
                idx_subg2 =  (dat['trial'].isin(trials))  &  (dat['phase'] > 1)  &  (dat['subject'] == s) &  (dat['valid'] == 1) &   (dat['suboptimal_goal_decision'] == -1)
                
                n_elements = np.nansum(idx)
                n_subg1 = np.nansum(idx_subg1)
                n_subg2 = np.nansum(idx_subg2)
                p_subg2[helper_idx,s_idx] = n_subg2/n_elements
                p_subg1[helper_idx,s_idx] = n_subg1/n_elements
                p_subg[helper_idx,s_idx]  = (n_subg2+n_subg1)/n_elements

            
    mean_subg2 = np.nanmean(p_subg2,1)
    std_subg2 = np.nanstd(p_subg2,1)
    mean_subg1 = np.nanmean(p_subg1,1)
    std_subg1 = np.nanstd(p_subg1,1)

    mean_subg = np.zeros(8)    
    mean_subg[range(0,8,2)]= mean_subg2
    mean_subg[range(1,8,2)]= mean_subg1
    std_subg = np.zeros(8)    
    std_subg[range(0,8,2)]= std_subg2
    std_subg[range(1,8,2)]= std_subg1
    
    #%% Time 
    helper_idx = -1
    p_subg2_time = np.zeros((nt,4,ns))
    p_subg1_time = np.zeros((nt,4,ns))
    p_subg_time = np.zeros((nt,4,ns))

    for j in range(2,-2,-1):
        helper_idx+=1
        s_idx = -1
        for s in subjects:
            s_idx += 1
            t_idx = -1 
            for t in trials:
                t_idx +=1
                if helper_idx < 3: 
                    idx = (dat['trial'] == t)   &   (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1) &  ~np.isnan(dat['suboptimal_goal_decision']) 
                    idx_subg1 =  (dat['trial'] == t)   &   (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) & (dat['valid'] == 1) &    (dat['suboptimal_goal_decision'] == -2)
                    idx_subg2 = (dat['trial'] == t)   &    (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  &  (dat['subject'] == s) &  (dat['valid'] == 1) &   (dat['suboptimal_goal_decision'] == -1)
                    
                    n_elements = np.nansum(idx)
                    n_subg1 = np.nansum(idx_subg1)
                    n_subg2 = np.nansum(idx_subg2)
                    p_subg2_time[t_idx,helper_idx,s_idx] = n_subg2/n_elements
                    p_subg1_time[t_idx,helper_idx,s_idx] = n_subg1/n_elements
                    p_subg_time[t_idx,helper_idx,s_idx]  = (n_subg2+n_subg1)/n_elements
            
                else:
                    idx =  (dat['trial'] == t)   &   (dat['phase'] > 1)   &  (dat['subject'] == s) & (dat['valid'] == 1) &  ~np.isnan(dat['suboptimal_goal_decision']) 
                    idx_subg1 = (dat['trial'] == t)   &    (dat['phase'] > 1)  &  (dat['subject'] == s) & (dat['valid'] == 1) &    (dat['suboptimal_goal_decision'] == -2)
                    idx_subg2 = (dat['trial'] == t)   &    (dat['phase'] > 1)  &  (dat['subject'] == s) &  (dat['valid'] == 1) &   (dat['suboptimal_goal_decision'] == -1)
                    
                    n_elements = np.nansum(idx)
                    n_subg1 = np.nansum(idx_subg1)
                    n_subg2 = np.nansum(idx_subg2)
                    p_subg2_time[t_idx,helper_idx,s_idx] = n_subg2/n_elements
                    p_subg1_time[t_idx,helper_idx,s_idx] = n_subg1/n_elements
                    p_subg_time[t_idx,helper_idx,s_idx]  = (n_subg2+n_subg1)/n_elements
            
    mean_subg2_time = np.nanmean(p_subg2_time,2)
    std_subg2_time = np.nanstd(p_subg2_time,2)
    mean_subg1_time = np.nanmean(p_subg1_time,2)
    std_subg1_time = np.nanstd(p_subg1_time,2)
    mean_subg_time = np.nanmean(p_subg_time,2)
    std_subg_time = np.nanstd(p_subg_time,2)


    if plotting == True:
        #%% Plotting
        if size == '1col':
          width  = 3.3  
          height = 0.8
        elif size == '2col':
          width  = 7
          height = 1.4 
        elif size == 'big':
          width  = 7*3
          height = 1.8*3
               
        barx = [0,1,3,4,6,7,9,10]
        bary = mean_subg
        barerr = std_subg
        
        
        ylabel = ['Suboptimal g-choice','Suboptimal g-choice','Suboptimal g2-choice','Suboptimal g1-choice']
        xlabel = ['','Trials','Trials','Trials']
        subplot_labels =['A','B','C','D']
        xticks = range(0,nt,2)
        xticklabels = range(1,nt+1,2)
        xticks2 = barx
        xticklabels2 = ['g2','g1','g2','g1','g2','g1','g2','g1']
        tick_length = 2
        tick_width = 1
        red_patch = mpatches.Patch(color='red', label='easy')
        green_patch = mpatches.Patch(color='green', label='medium')
        blue_patch = mpatches.Patch(color='blue', label='hard')
        grey_patch = mpatches.Patch(color='grey', label='all')
        colors =  ['red','green','blue','grey']
        ylim = (0,1)
        lw=1
        
        
        fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(width,height),gridspec_kw = {'width_ratios':[1, 1,1,1]})
        plt.tight_layout()
        
        for i in range(8):
            ax[0].bar(barx[i], bary[i], width=0.8, color = colors[(np.floor(i/2).astype(int))], yerr=barerr[i] ,error_kw=dict(elinewidth=1 ),alpha=0.5)
        
        ax[1].plot(mean_subg_time[:,3],color='black',linewidth=lw)
        ax[1].fill_between(range(nt),mean_subg_time[:,3]+std_subg_time[:,3], mean_subg_time[:,3]-std_subg_time[:,3], facecolor=colors[3], alpha=0.4)  
        
        for j in range(3):
            ax[2].plot(mean_subg2_time[:,j],color=colors[j],linewidth=lw)
            ax[2].fill_between(range(nt),mean_subg2_time[:,j]+std_subg2_time[:,j], mean_subg2_time[:,j]-std_subg2_time[:,j], facecolor=colors[j], alpha=0.2)
            ax[3].plot(mean_subg1_time[:,j],color=colors[j],linewidth=lw)
            ax[3].fill_between(range(nt),mean_subg1_time[:,j]+std_subg1_time[:,j], mean_subg1_time[:,j]-std_subg1_time[:,j], facecolor=colors[j], alpha=0.2)
        
    
        for i ,axes in enumerate(ax.flat):   
            axes.set_ylabel(ylabel[i],fontsize=7) 
            axes.set_xlabel(xlabel[i])
            axes.text(-0.15,1.2,subplot_labels[i],transform=axes.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')
        
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False) 
        
            axes.xaxis.set_tick_params(top='off', direction='out', width=1)
            axes.yaxis.set_tick_params(right='off', direction='out', width=1)
            axes.set_ylim(ylim)
            axes.tick_params(length=tick_length, width=tick_width)
    
            if i == 0:
               axes.legend(handles=[red_patch,green_patch,blue_patch,grey_patch],loc='upper center', bbox_to_anchor=(2.6, 1.6), ncol=4,frameon=False)
               axes.set_xticks(xticks2)
               axes.set_xticklabels(xticklabels2)
               axes.tick_params(axis='x',labelsize=6)
    #           axes.set_ylim((0,0.5))
               axes.set_ylim((0,0.7))
    
    
            else:
               axes.set_xticks(xticks)
               axes.set_xticklabels(xticklabels)
               
        
        if save == True:
            import time as time
            timestr = time.strftime("%Y%m%d-%H%M%S")
            fig.savefig('suboptimal_gc_'+timestr+'.png', dpi=300, bbox_inches='tight', transparent=True)

    if (len(trials) == 15) & (len(subjects)==89):
        df_A = pd.DataFrame({'sub_g1':p_subg1[0:3,:].flat.copy(),
                'sub_g2':p_subg2[0:3,:].flat.copy(),
                'subject':np.tile(np.arange(1,90),3),
                'condition':np.repeat(np.array([1,2,3]),89)})
        
        df_B = pd.DataFrame({'sub_g_time':p_subg_time[:,3,:].flat.copy(),
                'subject':np.tile(np.arange(1,90),1*15),
                'trial':np.repeat(np.arange(1,16),1*89)})      
                
        df_CD = pd.DataFrame({'sub_g1_time':p_subg1_time[:,0:3,:].flat.copy(),
                'sub_g2_time':p_subg2_time[:,0:3,:].flat.copy(),
                'subject':np.tile(np.arange(1,90),3*15),
                'condition':np.tile(np.repeat(np.array([1,2,3]),89),15),
                'trial':np.repeat(np.arange(1,16),3*89)})
    else:
        df_A = []
        df_B = []
        df_CD = []
    
    return df_A, df_B, df_CD, p_subg,p_subg_time,mean_subg,std_subg    
        
        
        
        
        
        
        
        
        
        
        