# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:39:55 2019

@author: ott
"""
import pandas as pd
import glob as glob

from tg_set_globalplotting import tg_set_globalplotting
from tg_preprocess_parameters import tg_preprocess_parameters  
from tg_untransform_params import tg_untransform_params

filename_group = glob.glob('../Results/theta_beta_gamma_kappa/group_posterior_sample*.csv')[0] 
filename_subjects = glob.glob('../Results/theta_beta_gamma_kappa/posterior_median_percentiles*.csv')[0] 
params_group_unc = pd.read_csv(filename_group)
params_group = tg_untransform_params(params_group_unc,kappamax=2) 
params = tg_preprocess_parameters(filename_subjects)   
agdat_multiple = pd.read_csv('../Results/preprocessed_optimal_agent.csv')
dat = pd.read_csv('../Results/preprocessed_results.csv')
tg_set_globalplotting(style='frontiers')

#%% Figure 3 
from tg_performance_accum_success import tg_performance_accum_success
tmp = tg_performance_accum_success(dat,agdat_multiple,save = False)    
    
#%% Figure 4
from tg_suboptimal import tg_suboptimal
tg_suboptimal(dat,save = False)   
    
#%% Figure 5    
from tg_plot_parameter import tg_plot_parameter
tg_plot_parameter(params,params_group,save = False)    
    
#%% Figure 6
from tg_reg_gc_param import tg_reg_gc_param
tg_reg_gc_param(dat,params,save = False)

#%% S5 Figure
from tg_ogv_signed import tg_ogv_signed
tg_ogv_signed(dat,save = False)

#%% S8 Figure
from tg_plot_selfreport_vs_params import tg_plot_selfreport_vs_params
tg_plot_selfreport_vs_params(dat,params,save = False)

#%% S12 Figure
from tg_plot_subg_hilobias import tg_plot_subg_hilobias
_=tg_plot_subg_hilobias(dat,params,save=False) 
