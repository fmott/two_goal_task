# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:04:33 2019

@author: ott
"""
import matplotlib.pyplot as plt

def tg_plot_parameter(params,params_group, size = '2col',save=False ):
   
    # Specifiy figure size
    if size == '1col':
      width  = 3.3  
      height = 0.7
    elif size == '2col':
      width  = 7
      height = 1.5 
    elif size == 'big':
      width  = 7*3
      height = 1.5*3
      
    # Assign paramters to new variables
    bias = params['bias_2_median'].values
    precision = params['beta_2_median'].values
    discount = params['gamma_median'].values
    kappa = params['kappa_median'].values
        
   # group parameters median and 95% CI
    g_bias_median = params_group['bias'].quantile(q=0.5)
    g_bias_95 = params_group['bias'].quantile(q=0.95)
    g_bias_5 = params_group['bias'].quantile(q=0.05)
       
    g_beta_median = params_group['beta_new'].quantile(q=0.5)
    g_beta_95 = params_group['beta_new'].quantile(q=0.95)
    g_beta_5 = params_group['beta_new'].quantile(q=0.05)
    
    g_gamma_median = params_group['gamma_new'].quantile(q=0.5)
    g_gamma_95 = params_group['gamma_new'].quantile(q=0.95)
    g_gamma_5 = params_group['gamma_new'].quantile(q=0.05)
    
    g_kappa_median = params_group['kappa_new'].quantile(q=0.5)
    g_kappa_95 = params_group['kappa_new'].quantile(q=0.95)
    g_kappa_5 = params_group['kappa_new'].quantile(q=0.05)
    
    #%% Plotting histogram 
    if size == '1col':
      width  = 3.3  
      height = 0.7
    elif size == '2col':
      width  = 7.4
      height = 1.5 
    elif size == 'big':
      width  = 7*3
      height = 1.5*3
    ylim =  [ [0,30], [0,30], [0,30] ,[0,30] ]
    xlabel = ['Strategy Preference'+r' ($\theta$)','Precision'+r' ($\beta$)','Discount'+r' ($\gamma$)','Reward Ratio'+r' ($\kappa$)']
    
    ylabel = 'Number of Subjects'
    xticks = [ [-1,0,1,2],[0,1,2,3,4,5,6,7], [ 0.92, 0.94,0.96, 0.98,1], [0.5,1,1.5,2]]
    xtick_labels = [[-1,0,1,2], [0,1,2,3,4,5,6,7],[0.92, 0.94,0.96, 0.98,1], [0.5,1,1.5,2] ]
    tick_length = 2
    tick_width = 1
    subplot_labels = ['A','B','C','D']
    col = 'grey'
    alph= 0.8
    fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(width,height))
    
    ax[0].hist(bias,color=col,alpha=alph)
    
    n,bins,patches = ax[1].hist(precision,color=col,alpha=alph,bins=15,range=(precision.min(),8))
    patches[-1].set_height(n[-1]+sum(precision > 8))
    ax[2].hist(discount,color=col,alpha=alph,bins=15)
    ax[3].hist(kappa,color=col,alpha=alph)

    # Median
    ax[0].axvline(x=g_bias_median,color='red',linewidth=1)
    ax[1].axvline(x=g_beta_median,color='red',linewidth=1)
    ax[2].axvline(x=g_gamma_median,color='red',linewidth=1)
    ax[3].axvline(x=g_kappa_median,color='red',linewidth=1)
    
    # 5% quantile
    ax[0].axvline(x=g_bias_5,color='red',linewidth=1,linestyle='--',alpha=0.5)
    ax[1].axvline(x=g_beta_5,color='red',linewidth=1,linestyle='--',alpha=0.5)
    ax[2].axvline(x=g_gamma_5,color='red',linewidth=1,linestyle='--',alpha=0.5)
    ax[3].axvline(x=g_kappa_5,color='red',linewidth=1,linestyle='--',alpha=0.5)
    
    # 95% quantile
    ax[0].axvline(x=g_bias_95,color='red',linewidth=1,linestyle='--',alpha=0.5)
    ax[1].axvline(x=g_beta_95,color='red',linewidth=1,linestyle='--',alpha=0.5)
    ax[2].axvline(x=g_gamma_95,color='red',linewidth=1,linestyle='--',alpha=0.5)
    ax[3].axvline(x=g_kappa_95,color='red',linewidth=1,linestyle='--',alpha=0.5)
    
    for i, axes in enumerate(ax.flat):
        axes.set_ylim(ylim[i])
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
            
        axes.xaxis.set_tick_params(top=False, direction='out', width=1)
        axes.yaxis.set_tick_params(right=False, direction='out', width=1)
            
        axes.set_xlabel(xlabel[i])

            
        axes.set_xticks(xticks[i])
        axes.set_xticklabels(xtick_labels[i])
        axes.tick_params(length=tick_length, width=tick_width)
        axes.text(-0.1,1.1,subplot_labels[i],transform=axes.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')
        if i == 0:
            axes.set_ylabel(ylabel)
 
    if save == True: 
        fig.savefig('params.png', dpi=300, bbox_inches='tight', transparent=True)
   
                