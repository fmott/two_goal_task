# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:46:39 2019

@author: ott
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.stats.descriptivestats as stats2

def tg_performance_accum_success(dat,agdat, save=False, size ='1col' ):

    ns = len(dat['subject'].unique())  
    helper_idx = -1
    threshold = 11 
    p_g1_success_1 = np.zeros((4,ns))
    p_g2_success_1 = np.zeros((4,ns))
    p_fail_1 = np.zeros((4,ns))
    accum_reward_1 = np.zeros((4,ns))
    for j in range(2,-2,-1):
        helper_idx+=1
        helper_idx2 = -1
        for s in range(ns):
            helper_idx2+=1
            
            if helper_idx < 3: 
                idx = (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  & (dat['trial'] == 15) &  (dat['subject'] == s+1)
                idx_g1 = (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  & (dat['trial'] == 15) &  (dat['subject'] == s+1)  &  ((dat['score_A_after'] >= threshold)  ^  (dat['score_B_after'] >= threshold))
                idx_g2 = (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  & (dat['trial'] == 15) &  (dat['subject'] == s+1)  &  ((dat['score_A_after'] >= threshold)  &  (dat['score_B_after'] >= threshold))
                idx_fail = (dat['phase'] > 1)  & ( (dat['start_condition'] == 1+2*j) |  (dat['start_condition'] == 2+2*j) )  & (dat['trial'] == 15) &  (dat['subject'] == s+1)  &  ((dat['score_A_after'] < threshold)  &  (dat['score_B_after'] < threshold))
        
                n_elements = np.nansum(idx) 
                n_g1_success = np.nansum(idx_g1) 
                n_g2_success = np.nansum(idx_g2) 
                n_fail = np.nansum(idx_fail) 
                p_g1_success_1 [helper_idx,helper_idx2] = n_g1_success / n_elements
                p_g2_success_1 [helper_idx,helper_idx2]  = n_g2_success / n_elements
                p_fail_1 [helper_idx,helper_idx2]  = n_fail / n_elements
                accum_reward_1 [helper_idx,helper_idx2] = n_g1_success*5 + n_g2_success*10
            else:
                idx = (dat['phase'] > 1)  & (dat['trial'] == 15) &  (dat['subject'] == s+1)
                idx_g1 = (dat['phase'] > 1)  & (dat['trial'] == 15) &  (dat['subject'] == s+1)  &  ((dat['score_A_after'] >= threshold)  ^  (dat['score_B_after'] >= threshold))
                idx_g2 = (dat['phase'] > 1)  & (dat['trial'] == 15) &  (dat['subject'] == s+1)  &  ((dat['score_A_after'] >= threshold)  &  (dat['score_B_after'] >= threshold))
                idx_fail = (dat['phase'] > 1)   & (dat['trial'] == 15) &  (dat['subject'] == s+1)  &  ((dat['score_A_after'] < threshold)  &  (dat['score_B_after'] < threshold))
        
                n_elements = np.nansum(idx) 
                n_g1_success = np.nansum(idx_g1) 
                n_g2_success = np.nansum(idx_g2) 
                n_fail = np.nansum(idx_fail) 
                p_g1_success_1 [helper_idx,helper_idx2] = n_g1_success / n_elements
                p_g2_success_1 [helper_idx,helper_idx2]  = n_g2_success / n_elements
                p_fail_1 [helper_idx,helper_idx2]  = n_fail / n_elements
                accum_reward_1 [helper_idx,helper_idx2] = n_g1_success*5 + n_g2_success*10
                
    
    mean_g1_success_1 = np.nanmean(p_g1_success_1,1)
    mean_g2_success_1 = np.nanmean(p_g2_success_1,1)
    mean_fail_1 = np.nanmean(p_fail_1,1)
    mean_accum_reward_1 = np.nanmean(accum_reward_1,1)
    std_g1_success_1 = np.nanstd(p_g1_success_1,1)
    std_g2_success_1 = np.nanstd(p_g2_success_1,1)
    std_fail_1 = np.nanstd(p_fail_1,1)
    std_accum_reward_1 = np.nanstd(accum_reward_1,1)
    
    ## Agent
    ns = len(agdat['subject'].unique())  
    helper_idx = -1
    threshold = 11 
    p_g1_success_2 = np.zeros((4,ns))
    p_g2_success_2 = np.zeros((4,ns))
    p_fail_2 = np.zeros((4,ns))
    accum_reward_2 = np.zeros((4,ns))
    for j in range(2,-2,-1):
        helper_idx+=1
        helper_idx2 = -1
        for s in range(ns):
            helper_idx2+=1
            
            if helper_idx < 3: 
                idx = (agdat['phase'] > 1)  & ( (agdat['start_condition'] == 1+2*j) |  (agdat['start_condition'] == 2+2*j) )  & (agdat['trial'] == 15) &  (agdat['subject'] == s+1)
                idx_g1 = (agdat['phase'] > 1)  & ( (agdat['start_condition'] == 1+2*j) |  (agdat['start_condition'] == 2+2*j) )  & (agdat['trial'] == 15) &  (agdat['subject'] == s+1)  &  ((agdat['score_A_after'] >= threshold)  ^  (agdat['score_B_after'] >= threshold))
                idx_g2 = (agdat['phase'] > 1)  & ( (agdat['start_condition'] == 1+2*j) |  (agdat['start_condition'] == 2+2*j) )  & (agdat['trial'] == 15) &  (agdat['subject'] == s+1)  &  ((agdat['score_A_after'] >= threshold)  &  (agdat['score_B_after'] >= threshold))
                idx_fail = (agdat['phase'] > 1)  & ( (agdat['start_condition'] == 1+2*j) |  (agdat['start_condition'] == 2+2*j) )  & (agdat['trial'] == 15) &  (agdat['subject'] == s+1)  &  ((agdat['score_A_after'] < threshold)  &  (agdat['score_B_after'] < threshold))
        
                n_elements = np.nansum(idx) 
                n_g1_success = np.nansum(idx_g1) 
                n_g2_success = np.nansum(idx_g2) 
                n_fail = np.nansum(idx_fail) 
                p_g1_success_2 [helper_idx,helper_idx2] = n_g1_success / n_elements
                p_g2_success_2 [helper_idx,helper_idx2]  = n_g2_success / n_elements
                p_fail_2 [helper_idx,helper_idx2]  = n_fail / n_elements
                accum_reward_2 [helper_idx,helper_idx2] = n_g1_success*5 + n_g2_success*10
    
            else:
                idx = (agdat['phase'] > 1)  & (agdat['trial'] == 15) &  (agdat['subject'] == s+1)
                idx_g1 = (agdat['phase'] > 1)  & (agdat['trial'] == 15) &  (agdat['subject'] == s+1)  &  ((agdat['score_A_after'] >= threshold)  ^  (agdat['score_B_after'] >= threshold))
                idx_g2 = (agdat['phase'] > 1)  & (agdat['trial'] == 15) &  (agdat['subject'] == s+1)  &  ((agdat['score_A_after'] >= threshold)  &  (agdat['score_B_after'] >= threshold))
                idx_fail = (agdat['phase'] > 1)   & (agdat['trial'] == 15) &  (agdat['subject'] == s+1)  &  ((agdat['score_A_after'] < threshold)  &  (agdat['score_B_after'] < threshold))
        
                n_elements = np.nansum(idx) 
                n_g1_success = np.nansum(idx_g1) 
                n_g2_success = np.nansum(idx_g2) 
                n_fail = np.nansum(idx_fail) 
                p_g1_success_2 [helper_idx,helper_idx2] = n_g1_success / n_elements
                p_g2_success_2 [helper_idx,helper_idx2]  = n_g2_success / n_elements
                p_fail_2 [helper_idx,helper_idx2]  = n_fail / n_elements
                accum_reward_2 [helper_idx,helper_idx2] = n_g1_success*5 + n_g2_success*10
    
    
    mean_g1_success_2 = np.nanmean(p_g1_success_2,1)
    mean_g2_success_2 = np.nanmean(p_g2_success_2,1)
    mean_fail_2 = np.nanmean(p_fail_2,1)
    mean_accum_reward_2 = np.nanmean(accum_reward_2,1)
    
    std_g1_success_2 = np.nanstd(p_g1_success_2,1)
    std_g2_success_2 = np.nanstd(p_g2_success_2,1)
    std_fail_2 = np.nanstd(p_fail_2,1)
    std_accum_reward_2 = np.nanstd(accum_reward_2,1)
    
    ## Difference between groups (Subject - Agent)
    mean_diff_g1_success = mean_g1_success_1 - mean_g1_success_2
    mean_diff_g2_success = mean_g2_success_1 - mean_g2_success_2
    mean_diff_fail = mean_fail_1 - mean_fail_2
    mean_diff_accum = mean_accum_reward_1 - mean_accum_reward_2
    std_diff_g1_success = std_g1_success_1 + std_g1_success_2
    std_diff_g2_success = std_g2_success_1 + std_g2_success_2
    std_diff_fail = std_fail_1 + std_fail_2
    std_diff_accum = std_accum_reward_1 + std_accum_reward_2
    
    #%% Plotting Accumulated Reward
    if size == '1col':
      width  = 3.5  
      height = 4
    elif size == '2col':
      width  = 7
      height = 10 
    elif size == 'big':
      width  = 7*3
      height = 10*3
      
        
    # Signtest
    M_accum=np.zeros((4,1)) 
    psign_accum=np.zeros((4,1))
    for i in range(4):
        M_accum[i],psign_accum[i] = stats2.sign_test(accum_reward_1[i,:] - accum_reward_2[i,0])
           
    barx = np.array([1, 2, 3, 4])

    bary_1 = np.ravel(mean_accum_reward_1)
    bary_2 = np.ravel(mean_accum_reward_2)
    bary_diff =  np.ravel(mean_diff_accum)
    
    barerr_1 = np.ravel(std_accum_reward_1)
    barerr_2 = np.ravel(std_accum_reward_2)
    barerr_diff =  np.ravel(std_diff_accum)
    
    # Plotting specs 
    my_colors = np.array( ['red','green','blue','grey'])
    ylabel = 'Total Reward [Cents]'
    titles = ['Subjects','Optimal agent', 'Difference (subject-agent)'] 
    subplot_labels =['A','C','E']

    xticks = barx
    xticklabels = ['easy','medium','hard','all']
    red_patch = mpatches.Patch(color='red', label='easy')
    green_patch = mpatches.Patch(color='green', label='medium')
    blue_patch = mpatches.Patch(color='blue', label='hard')
    grey_patch = mpatches.Patch(color='grey', label='all')
    ylim=(-40,40)
    tick_length = 2
    tick_width = 1
    alph = 0.5
    
    fig, ax = plt.subplots(nrows=3,ncols=2,figsize = (width,height),gridspec_kw = {'width_ratios':[1,3]})
    plt.tight_layout()
    for i in range(np.size(barx)):
        ax.flat[0].bar(barx[i], bary_1[i], width=0.7, color = my_colors[i], yerr=barerr_1[i] ,error_kw=dict(elinewidth=1 ),alpha=alph)
        ax.flat[2].bar(barx[i], bary_2[i], width=0.7, color = my_colors[i], yerr=barerr_2[i],error_kw=dict(elinewidth=1 ),alpha=alph)
        ax.flat[4].bar(barx[i], bary_diff[i], width=0.7, color = my_colors[i], yerr=barerr_diff[i],error_kw=dict(elinewidth=1 ),alpha=alph)
        if (psign_accum[i] <= 0.05) & (psign_accum[i] > 0.01):
            ax.flat[4].text(barx[i],bary_diff[i]-barerr_diff[i]-10,'*',fontsize=6,fontweight='bold',horizontalalignment='center',verticalalignment='center')
        elif (psign_accum[i] <= 0.01) & (psign_accum[i] > 0.001):
            ax.flat[4].text(barx[i],bary_diff[i]-barerr_diff[i]-10,'**',fontsize=6,fontweight='bold',horizontalalignment='center',verticalalignment='center')
        elif (psign_accum[i] <= 0.001):
            ax.flat[4].text(barx[i],bary_diff[i]-barerr_diff[i]-10,'***',fontsize=6,fontweight='bold',horizontalalignment='center',verticalalignment='center')
            
    for i ,axes in enumerate([ax.flat[0],ax.flat[2],ax.flat[4]]):
        
        axes.set_xticks(xticks)
        axes.set_xticklabels([])
        axes.text(-0.1,1.12,subplot_labels[i],transform=axes.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False) 
        axes.xaxis.set_tick_params(top='off', direction='out', width=1)
        axes.yaxis.set_tick_params(right='off', direction='out', width=1)
        axes.set_ylabel(ylabel,fontsize=8,labelpad=None) 

        if i == 0:
            axes.legend(handles=[red_patch,green_patch,blue_patch,grey_patch],loc='upper center', bbox_to_anchor=(2.2, 1.75), ncol=4,frameon=False)
        
        if i < 2:
            axes.tick_params(length=tick_length, width=tick_width)
            axes.text(3.3,1.30,titles[i],transform=axes.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=9)
            axes.set_ylim((0,450))

        else:
            axes.tick_params(length=0, width=0,axis='x')
            axes.tick_params(length=tick_length, width=tick_width)

        if i == 2:
            axes.axhline(0,linewidth = 0.5, color = 'black')
            axes.xaxis.set_tick_params(bottom='off')
            axes.spines['bottom'].set_visible(False) 
            axes.set_ylim(ylim)
   
    # Sign test 
    M_g1=np.zeros((4,1))
    psign_g1=np.zeros((4,1))
    M_g2=np.zeros((4,1))
    psign_g2=np.zeros((4,1))
    M_fail=np.zeros((4,1))
    psign_fail=np.zeros((4,1))
    for i in range(4):
        M_g1[i],psign_g1[i] = stats2.sign_test( p_g1_success_1[i,:] - p_g1_success_2[i,0]  )
        M_g2[i],psign_g2[i] = stats2.sign_test( p_g2_success_1[i,:] - p_g2_success_2[i,0]  )
        M_fail[i],psign_fail[i] = stats2.sign_test( p_fail_1[i,:] - p_fail_2[i,0]  )
    
    # Additional tests 
    M_g2_easy_vs_medium, psign_g2_easy_vs_medium = stats2.sign_test( p_g2_success_1[0,:],p_g2_success_1[1,:]  )

    # data to be plotted 
    barx = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
    
    bary_1 = np.ravel(np.column_stack((mean_g1_success_1,mean_g2_success_1,mean_fail_1)))
    bary_2 = np.ravel(np.column_stack((mean_g1_success_2,mean_g2_success_2,mean_fail_2)))
    bary_diff =  np.ravel(np.column_stack((mean_diff_g1_success,mean_diff_g2_success,mean_diff_fail)))
    
    barerr_1 = np.ravel(np.column_stack((std_g1_success_1,std_g2_success_1,std_fail_1)))
    barerr_2 = np.ravel(np.column_stack((std_g1_success_2,std_g2_success_2,std_fail_2)))
    barerr_diff =  np.ravel(np.column_stack((std_diff_g1_success,std_diff_g2_success,std_diff_fail)))
    
    # plotting specs
    bar_p = np.ravel(np.column_stack((psign_g1,psign_g2,psign_fail)))
    tmp = barerr_diff.copy()
    tmp[bary_diff < 0 ] = 0
    tmp2 = bary_diff.copy()
    tmp2[bary_diff < 0 ] = 0
    bar_ppos = tmp2+tmp + 0.02 
    my_colors = np.array( ['red','red','red','green','green','green','blue','blue','blue','grey','grey','grey'])  
    ylabel = 'Proportion Success'
    titles = ['Subjects','Optimal agent', 'Difference (subject - agent)'] 
    subplot_labels =['B','D','F']
    xticks = barx
    xticklabels = ['G1','G2','fail','G1','G2','fail','G1','G2','fail','G1','G2','fail']
    red_patch = mpatches.Patch(color='red', label='easy')
    green_patch = mpatches.Patch(color='green', label='medium')
    blue_patch = mpatches.Patch(color='blue', label='hard')
    grey_patch = mpatches.Patch(color='grey', label='all')
    tick_length = 2
    tick_width = 1
    alph = 0.5 
    yticks= np.arange(0,1.2,0.2)
    yticklabels= np.round(np.arange(0,1.2,0.2),1)
    
    for i in range(np.size(barx)):
        ax.flat[1].bar(barx[i], bary_1[i], width=0.8, color = my_colors[i], yerr=barerr_1[i] ,error_kw=dict(elinewidth=1 ),alpha=alph)
        ax.flat[3].bar(barx[i], bary_2[i], width=0.8, color = my_colors[i], yerr=barerr_2[i],error_kw=dict(elinewidth=1 ),alpha=alph)
        ax.flat[5].bar(barx[i], bary_diff[i], width=0.8, color = my_colors[i], yerr=barerr_diff[i],error_kw=dict(elinewidth=1 ),alpha=alph)
        if (bar_p[i] <= 0.05) & (bar_p[i] > 0.01):
            ax.flat[5].text(barx[i], bar_ppos[i],'*',fontsize=6,fontweight='bold',horizontalalignment='center',verticalalignment='center')
        elif (bar_p[i] <= 0.01) & (bar_p[i] > 0.001):
            ax.flat[5].text(barx[i],bar_ppos[i],'**',fontsize=6,fontweight='bold',horizontalalignment='center',verticalalignment='center')
        elif (bar_p[i] <= 0.001):
            ax.flat[5].text(barx[i],bar_ppos[i],'***',fontsize=6,fontweight='bold',horizontalalignment='center',verticalalignment='center')
            
    for i ,axes in enumerate([ax.flat[1],ax.flat[3],ax.flat[5]]):
        
        axes.set_xticks(xticks)
        axes.set_xticklabels(xticklabels)
        axes.text(-0.1,1.12,subplot_labels[i],transform=axes.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')
        axes.tick_params(axis='x',labelsize=5)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False) 
        axes.xaxis.set_tick_params(top='off', direction='out', width=1)
        axes.yaxis.set_tick_params(right='off', direction='out', width=1)
        axes.set_ylabel(ylabel,fontsize=8) 

        if i < 2:
            axes.tick_params(length=tick_length, width=tick_width)
            axes.set_ylim((0,1))
            axes.set_yticks(yticks)
            axes.set_yticklabels(yticklabels)
            axes.text(-0.5,1.30,titles[i],transform=axes.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=9)

        else:
            axes.tick_params(length=tick_length, width=tick_width)
            axes.tick_params(length=0, width=0,axis='x')

    
        if i == 2:
            axes.axhline(0,linewidth = 0.5, color = 'black')
            axes.xaxis.set_tick_params(bottom='off')
            axes.spines['bottom'].set_visible(False) 
            axes.set_ylabel(ylabel,fontsize=8,labelpad=-2) 
            axes.text(0,1.3,titles[i],transform=axes.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=9)

        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.7)
        
        if save == True:
            fig.savefig('performance.png', dpi=300, bbox_inches='tight', transparent=True)
            
        return_dict={'m_accum': M_accum,
                     'p_accum': psign_accum,
                     'm_g1': M_g1,
                     'p_g1':psign_g1,
                     'm_g2':M_g2,
                     'p_g2':psign_g2,
                     'm_fail':M_fail,
                     'p_fail':psign_fail}
        
    return return_dict