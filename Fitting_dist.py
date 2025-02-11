# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:59:01 2025

@author: KPFS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nabqr as nq
import seaborn as sns
import scipy.stats as stats
from statsmodels.tsa.api import acf

#i think this is bastians quantiles
quantiles = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

#%%

ensembles = pd.read_pickle("data/ensembles_input_all/DK2_offshore_wind_power_oct24.pkl")
#strip timezone from ensembles
ensembles.index = ensembles.index.to_series().apply(lambda x: x.tz_localize(None)).values
ensembles = ensembles[~ensembles.index.duplicated()]
obs = pd.read_pickle("data/observations/actuals_DK2_offshore_wind_power_oct24.pkl")
obs = obs.dropna()

#%%
corrected, estimated_quantiles, actual, beta, orig  = nq.pipeline(ensembles, obs.values, epochs = 20, quantiles_taqr = quantiles)
estimated_quantiles.columns = quantiles
#%%
"""
def piecewise_constant_quantile(x, quantile_est):
    tmp = x > quantile_est
    if tmp.any(): 
        return quantiles[tmp][-1]
    else:
        return quantiles[0]

q = [piecewise_constant_quantile(actual[i], estimated_quantiles.iloc[i,:]) for i in range(len(actual))]

plt.figure(figsize = (14,8))
plt.subplot(1,2,1)
plt.scatter(list(range(len(q))),q)
plt.subplot(1,2,2)
sns.histplot(x = q, bins =  np.insert(quantiles, [0,len(quantiles)], [0,1]), stat = "density")
"""
#%%

def evaluate_estimator(actual, est_quantiles, estimator, discrete = False):
    X = actual.index
    dist = stats.norm()
    x_dist = np.linspace(-3,3,1000)
    dist_quantiles = dist.ppf(np.arange(1,len(X)+1) / (len(X)+1))
    
    resids = np.zeros(len(X))
    
    
    
    for i in range(len(X)):
        
        estimator.fit(est_quantiles.iloc[i,:])
        resids[i] = estimator.estimate(actual.iloc[i])
    resids = dist.ppf(resids)
    
    ## plot
    lims = dist.ppf([0.001/len(X), 0.01, 0.05, 0.5, 0.95, 0.99, 1 - 0.001/len(X)])
    
    fig, axes = plt.subplots(2,2, figsize = (14,8))
    
    sns.scatterplot(x = X, y = resids, ax = axes[0,0])
    axes[0,0].hlines(lims, X[0], X[-1], color = 'k', linestyle = '--')
    
    sns.histplot(x = resids, ax = axes[0,1], discrete = discrete, stat = "density")
    axes[0,1].plot(x_dist, dist.pdf(x_dist), color = 'k')
    
    axes[1,0].scatter(dist_quantiles, sorted(resids))
    tmp = np.linspace(min(dist_quantiles), max(dist_quantiles),10)
    axes[1,0].plot(tmp,tmp)
    
    
    a, c = acf(resids, alpha = 0.05, nlags = 50)
    axes[1,1].bar(np.arange(0,len(a)),a)
    axes[1,1].fill_between(np.arange(0,len(a)), c[:,0] - a, c[:,1] - a, color = 'black', alpha = 0.3)
    
    return resids, axes
    
class quantile_estimator():
    def __init__(self, quantiles = [0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99]):
        self.quantiles = np.array(quantiles[:])
        
    def fit(self, quantiles):
        return
    def estimate(self, y):
        return

class constant_estimator(quantile_estimator):
    
    def fit(self, est_quantiles):
        self.q = est_quantiles[:]
        
    def estimate(self, y):
        tmp = y > self.q
        if tmp.any(): 
            return self.quantiles[tmp][-1]
        else:
            return self.quantiles[0]
     
class piecewise_linear(quantile_estimator):
    
    def __init__(self, quantiles, max_val, min_val):
        
        super().__init__(quantiles)
        
        tmp = np.zeros(len(self.quantiles)+2)
        tmp[1:-1] = self.quantiles
        tmp[0] = 0
        tmp[-1] = 1
        
        self.quantiles = tmp
        
        self.max_val = max_val
        self.min_val = min_val
    
    
    def fit(self, est_quantiles):
        self.est_quantiles = est_quantiles
        
        self.q_vals = np.zeros(len(est_quantiles)+2)
        self.q_vals[1:-1] = est_quantiles
        
        self.q_vals[-1] = self.max_val
        self.q_vals[0] = self.min_val

        self.diffs = self.q_vals[1:] - self.q_vals[:-1]
        self.coefs = self.quantiles[1:] - self.quantiles[:-1]
    
    def estimate(self, y):
        
        conds = (y >= self.q_vals[:-1]) & (y < self.q_vals[1:])
        vals = self.coefs * (y - self.q_vals[:-1]) / self.diffs + self.quantiles[:-1]

        
        return np.piecewise(y, conds, vals)
        
        
        
        

#%% 

est1 = constant_estimator(quantiles)
evaluate_estimator(actual, estimated_quantiles, est1)

#%%
est2 = piecewise_linear(quantiles, actual.max()*1.01, actual.min()-1)
evaluate_estimator(actual,estimated_quantiles, est2)
