"""This module contains the class that defines the interaction between
different modules that govern the agent's behavior.
"""

from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch import ones, zeros
from torch.distributions import constraints, biject_to

import pyro.distributions as dist
from pyro import sample, param, clear_param_store, plate
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions.util import sum_rightmost

class Inferrer(object):
    
    def __init__(self, agent, stimulus, response_data, mask = None, fixed_params = None):
        
        self.agent = agent # agent used for computing response probabilities
        self.stimulus = stimulus # stimulus and action outcomes presented to each participant
        self.rdata = response_data # measured behavioral data accross all subjects
        self.runs, self.nb, self.nt = self.rdata.shape
        
        # set a mask for excluding certain responses (e.g. NaN resonses) from 
        # the computations of the log-model evidence and posterior beleifs over
        # parameter values
        if mask is not None:
            self.notnans = mask
            self.mask = mask.float()
        else:
            self.notnans = ones(self.rdata.shape, dtype=torch.bool)
            self.mask = ones(self.rdata.shape)
            
        if fixed_params is not None:
            n_fixed = len(fixed_params['labels'])
            self.npars = agent.npars - n_fixed
            
            self.locs = {}
            self.locs['fixed'] = fixed_params['labels']
            self.locs['free'] = list(set(range(agent.npars)) - \
                                set(fixed_params['labels']))
            self.values = fixed_params['values']
            self.fixed_values = True
        else:
            self.npars = agent.npars
            self.fixed_values = False
    
    def model_non_hier(self):
        """
        Generative model of behavioral responses. The prior is defined as a 
        normal distribution with large uncertainty.
        """
        n = self.runs #number of subjects
        npars = self.npars #number of parameters
        
        m = param('m', zeros(npars))
        s = param('s', ones(npars), constraint=constraints.positive)

        # define prior mean over model parameters and subjects
        with plate('subjects', n):
            x = sample('x', dist.Normal(m, s).to_event(1))
        
        if self.fixed_values:
            trans_pars = zeros(n, self.agent.npars)
            trans_pars[:, self.locs['fixed']] = self.values
            trans_pars[:, self.locs['free']] = x
        else:
            trans_pars = x

        agent = self.agent.set_parameters(trans_pars)
        for b in range(self.nb):
            for t in range(self.nt):
                #update single trial
                states = self.stimulus['states'][:, b, t]
                offers = self.stimulus['offers'][:, b, t]
                
                self.agent.update_beliefs(b, t, states, offers)
                self.agent.planning(b, t)
        
        logits = agent.logprobs[self.notnans]      
        responses = self.rdata[self.notnans]
        
        with plate('observations', len(responses)):
            sample('obs', dist.Categorical(logits=logits), obs=responses)

    def model_cent_hs(self):
        """
        Generative model of behavioral responses. The prior is defined as a 
        centered horseshoe prior.
        """
        n = self.runs #number of subjects
        npars = self.npars #number of parameters

        # define hyper priors over model parameters.

        # define prior uncertanty over model parameters and subjects
        lam = param('lam', ones(1), constraint=constraints.positive)
        sigma_g = sample('sigma_g', dist.HalfCauchy(ones(npars)).to_event(1))
        
        # each model parameter has a hyperprior defining group level mean
        m = param('m', zeros(npars))
        s = param('s', ones(npars), constraint=constraints.positive)
        mu_g=sample('mu_g',dist.Normal(m, s).to_event(1))        

        # define prior mean over model parametrs and subjects
        with plate('subjects', n):
                x = sample('x', dist.Normal(mu_g, lam * sigma_g).to_event(1))
        
        if self.fixed_values:
            trans_pars = zeros(n, self.agent.npars)
            trans_pars[:, self.locs['fixed']] = self.values
            trans_pars[:, self.locs['free']] = x
        else:
            trans_pars = x

        agent = self.agent.set_parameters(trans_pars)
        for b in range(self.nb):
            for t in range(self.nt):
                #update single trial
                states = self.stimulus['states'][:, b, t]
                offers = self.stimulus['offers'][:, b, t]
                
                self.agent.update_beliefs(b, t, states, offers)
                self.agent.planning(b, t)
        
        logits = agent.logprobs[self.notnans]      
        responses = self.rdata[self.notnans]
        
        with plate('observations', responses.shape[0]):
            sample('obs', dist.Categorical(logits=logits), obs=responses)
            
    def model_cent_hsp(self):
        """
        Generative model of behavioral responses. The prior is defined as a 
        centered horseshoe plus prior.
        """
        n = self.runs #number of subjects
        npars = self.npars #number of parameters

        # define hyper priors over model parameters.
        # each model parameter has a hyperpriors defining group level mean
        m = param('m', zeros(npars))
        s = param('s', ones(npars), constraint=constraints.positive)
        mu_g=sample('mu_g',dist.Normal(m, s).to_event(1))

        # define prior uncertanty over model parameters and subjects
        lam = param('lam', ones(1), constraint=constraints.positive)
        sigma_g = sample('sigma_g', dist.HalfCauchy(1.).expand([npars]).to_event(1))
        
        # define prior mean over model parametrs and subjects
        gam = param('gam', ones(n, 1), constraint=constraints.positive)
        with plate('subjects', n):
            # define prior uncertainty over both parameters and subjects
            sigma_x = sample('sigma_x', dist.HalfCauchy(1.).expand([npars]).to_event(1))
            x = sample('x', dist.Normal(mu_g, lam * gam * sigma_x * sigma_g).to_event(1))
        
        if self.fixed_values:
            trans_pars = zeros(n, self.agent.npars)
            trans_pars[:, self.locs['fixed']] = self.values
            trans_pars[:, self.locs['free']] = x
        else:
            trans_pars = x

        agent = self.agent.set_parameters(trans_pars)
        for b in range(self.nb):
            for t in range(self.nt):
                #update single trial
                states = self.stimulus['states'][:, b, t]
                offers = self.stimulus['offers'][:, b, t]
                
                self.agent.update_beliefs(b, t, states, offers)
                self.agent.planning(b, t)
        
        logits = agent.logprobs[self.notnans]      
        responses = self.rdata[self.notnans]
        
        with plate('observations'):
            sample('obs', dist.Categorical(logits=logits), obs=responses)
        
    def guide_non_hier(self):
        """
        Approximate posterior for the horseshoe prior. We assume posterior in the form 
        of the multivariate normal distriburtion for the global mean and standard deviation
        and multivariate normal distribution for the parameters of each subject independently.
        """
        nsub = self.runs #number of subjects
        npar = self.npars #number of parameters
                
        m_locs = param('m_locs', zeros(nsub, npar))
        st_locs = param('s_locs', torch.eye(npar).repeat(nsub, 1, 1), 
                   constraint=constraints.lower_cholesky)

        with plate('subjects', nsub):
            x = sample("x", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
        
        return {'x': x}    
    
    def guide_horseshoe(self):
        """
        Approximate posterior for the horseshoe prior. We assume posterior in the form 
        of the multivariate normal distriburtion for the global mean and standard deviation
        and multivariate normal distribution for the parameters of each subject independently.
        """
        nsub = self.runs #number of subjects
        npar = self.npars #number of parameters
        trns = biject_to(constraints.positive)

        
        m_hyp = param('m_hyp', zeros(2*npar))
        st_hyp = param('scale_tril_hyp', 
                              torch.eye(2*npar), 
                              constraint=constraints.lower_cholesky)
        hyp = sample('hyp', dist.MultivariateNormal(m_hyp, 
                                                  scale_tril=st_hyp), 
                            infer={'is_auxiliary': True})
        
        unc_mu = hyp[:npar]
        unc_sigma = hyp[npar:]
    
    
        c_sigma = trns(unc_sigma)
    
        ld_sigma = trns.inv.log_abs_det_jacobian(c_sigma, unc_sigma)
        ld_sigma = sum_rightmost(ld_sigma, ld_sigma.dim() - c_sigma.dim() + 1)
    
        mu_g = sample("mu_g", dist.Delta(unc_mu, event_dim=1))
        sigma_g = sample("sigma_g", dist.Delta(c_sigma, log_density=ld_sigma, event_dim=1))
        
        m_locs = param('m_locs', zeros(nsub, npar))
        st_locs = param('s_locs', torch.eye(npar).repeat(nsub, 1, 1), 
                   constraint=constraints.lower_cholesky)

        with plate('subjects', nsub):
            x = sample("x", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
        
        return {'mu_g': mu_g, 'sigma_g': sigma_g, 'x': x}
    
    def guide_horseshoe_plus(self):
        
        npar = self.npars  # number of parameters
        nsub = self.runs  # number of subjects
        trns = biject_to(constraints.positive)

        
        m_hyp = param('m_hyp', zeros(2*npar))
        st_hyp = param('scale_tril_hyp', 
                              torch.eye(2*npar), 
                              constraint=constraints.lower_cholesky)
        hyp = sample('hyp', dist.MultivariateNormal(m_hyp, 
                                                  scale_tril=st_hyp), 
                            infer={'is_auxiliary': True})
        
        unc_mu = hyp[:npar]
        unc_sigma = hyp[npar:]
    
    
        c_sigma = trns(unc_sigma)
    
        ld_sigma = trns.inv.log_abs_det_jacobian(c_sigma, unc_sigma)
        ld_sigma = sum_rightmost(ld_sigma, ld_sigma.dim() - c_sigma.dim() + 1)
    
        mu_g = sample("mu_g", dist.Delta(unc_mu, event_dim=1))
        sigma_g = sample("sigma_g", dist.Delta(c_sigma, log_density=ld_sigma, event_dim=1))
        
        m_tmp = param('m_tmp', zeros(nsub, 2*npar))
        st_tmp = param('s_tmp', torch.eye(2*npar).repeat(nsub, 1, 1), 
                   constraint=constraints.lower_cholesky)

        with plate('subjects', nsub):
            tmp = sample('tmp', dist.MultivariateNormal(m_tmp, 
                                                  scale_tril=st_tmp), 
                            infer={'is_auxiliary': True})
            
            unc_locs = tmp[..., :npar]
            unc_scale = tmp[..., npar:]
            
            c_scale = trns(unc_scale)
            
            ld_scale = trns.inv.log_abs_det_jacobian(c_scale, unc_scale)
            ld_scale = sum_rightmost(ld_scale, ld_scale.dim() - c_scale.dim() + 1)
            
            x = sample("x", dist.Delta(unc_locs, event_dim=1))
            sigma_x = sample("sigma_x", dist.Delta(c_scale, log_density=ld_scale, event_dim=1))
        
        return {'mu_g': mu_g, 'sigma_g': sigma_g, 'sigma_x': sigma_x, 'x': x}
            
    def infer_posterior(self,
                        iter_steps=1000,
                        num_particles=10,
                        optim_kwargs={'lr':.1},
                        parametrisation='cent-horseshoe'):

        clear_param_store()
        
        #mean-field approximation to the posterior depending on the prior type
        if parametrisation == 'non-hierarchical':
            model = self.model_non_hier
            guide = self.guide_non_hier
        elif parametrisation == 'cent-horseshoe':
            model = self.model_cent_hs
            guide = self.guide_horseshoe
        elif parametrisation == 'cent-horseshoe+':
            model = self.model_cent_hsp
            guide = self.guide_horseshoe_plus
        else:
            print('Error: model not specified')
            model = None
            guide = None

        svi = SVI(model=model,
                  guide=guide,
                  optim=Adam(optim_kwargs),
                  loss=Trace_ELBO(num_particles=num_particles))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:
            loss.append(svi.step())
            pbar.set_description("Mean ELBO %6.2f" % torch.Tensor(loss[-20:]).mean())
        
        try:
            self.mean = param('auto_loc')
            self.std = param('auto_scale')
        except:
            pass
        
        self.guide = guide        
        self.loss = loss

    def sample_from_posterior(self, labels, centered=True, n_samples=10000):
        
        n = self.runs
        npars = self.npars
        assert npars == len(labels)
        
        keys = ['x', 'sigma_x', 'mu_g', 'sigma_g']
        
        trans_pars = np.zeros((n_samples, n, npars))
        sigma_trans_pars = np.zeros((n_samples, n, npars))
        
        group = np.zeros((n_samples, npars))
        sigma_group = np.zeros((n_samples, npars))
        
        for i in range(n_samples):
            sample = self.guide()
            for key in keys:
                sample.setdefault(key, ones(1))
                
            mu_g = sample['mu_g']
            sigma_g = sample['sigma_g']
            sigma_x = sample['sigma_x']
            if centered:
                x = sample['x']
            else:
                x = sample['x'] * sigma_g * sigma_x + mu_g
            
            trans_pars[i] = x.detach().numpy()
            sigma_trans_pars[i] = sigma_x.detach().numpy()
            
            group[i] = mu_g.detach().numpy()
            sigma_group[i] = sigma_g.detach().numpy()
        
        subject_label = np.tile(range(1, n+1), (n_samples, 1)).reshape(-1)
        tp_df = pd.DataFrame(data=trans_pars.reshape(-1, npars), columns=labels)
        tp_df['subject'] = subject_label
        stp_df = pd.DataFrame(data=sigma_trans_pars.reshape(-1, npars), columns=labels)
        stp_df['subject'] = subject_label
        
        g_df = pd.DataFrame(data=group, columns=labels)
        sg_df = pd.DataFrame(data=sigma_group, columns=labels)
        
        return (tp_df, stp_df, g_df, sg_df)
    
    def format_results(self, labels, centered=True, n_samples=10000):
        """Sample from posterior and map to constrained parameter values.
           Returns median, 5th and 95th percentile for each parameter and subject. 
        """
        from numpy import percentile
        n = self.runs
        npars = self.agent.npars
        
        var_names = ['x', 'sigma_x', 'sigma_g', 'mu_g']
        par_names = ['beta', 'bias', 'gamma', 'kappa']
        par_values = {}
        
        for i in range(n_samples):
            sample = self.guide()
            if centered:
                x = sample['x']
            else:
                for var in var_names:
                    sample.setdefault(var, ones(1))
                    
                x = sample['x'] * sample['sigma_g'] * sample['sigma_x'] + sample['mu_g']
            if self.fixed_values:
                trans_pars = zeros(n, npars)
                trans_pars[:, self.locs['fixed']] = self.values
                trans_pars[:, self.locs['free']] = x.detach()
            else:
                trans_pars = x.detach()
            
            self.agent.set_parameters(trans_pars)
            
            for name in par_names:
                par_values.setdefault(name, [])
                par_values[name].append(getattr(self.agent, name))
        
        count = {}
        percentiles = {}
        for name in par_names:
            count.setdefault(name, 0)
            par_values[name] = torch.stack(par_values[name], dim=0)
            if len(par_values[name].shape) < 3:
                par_values[name] = par_values[name].unsqueeze(-1)
            for lbl in labels:
                if lbl.startswith(name):
                    values = par_values[name][..., count[name]].numpy()
                    count[name] += 1
                    percentiles[lbl] = percentile(values, [5, 50, 95], axis=0).reshape(-1)
        
        df_percentiles = pd.DataFrame(percentiles)
        
        subjects = np.tile(np.arange(1, n+1), [3, 1]).reshape(-1)
        df_percentiles['subjects'] = subjects
        variables = np.tile(np.array(['5th', 'median', '95th']), [n, 1]).T.reshape(-1)
        df_percentiles['variables'] = variables
        
        return df_percentiles.melt(id_vars=['subjects', 'variables'], var_name='parameter')
        

    def compute_model_attribution(self):
    	"""Return subject specific model evidence"""
    	#TODO: Figure out how to implement this
    	pass