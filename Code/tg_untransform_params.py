# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:54:31 2019

@author: ott
"""

import numpy as np
from tg_sigmoid import tg_sigmoid
# Un-Transform parameters
# beta: np.exp(beta)
# bias: bias
# gamma: sigmoid(gamma)
# kappa: sigmoid(kappa)*2

def tg_untransform_params(params,kappamax):
    pars = np.isin(['beta','bias','gamma','kappa'],params.columns)
    
    if pars[0]:
        params['beta_new'] = np.exp(params['beta'])
    if pars[2]:
        params['gamma_new'] = tg_sigmoid(params['gamma'])
    if pars[3]:
        params['kappa_new'] = tg_sigmoid(params['kappa'])*kappamax
    
    return params 