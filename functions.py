# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:49:21 2025

@author: KPFS
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from statsmodels.tsa.api import acf



def evaluate_estimator(actual, est_quantiles, estimator, discrete = False):
    X = actual.index
    dist = stats.norm()
    
    x_dist = np.linspace(-3,3,1000)
    dist_quantiles = dist.ppf(np.arange(1,len(X)+1) / (len(X)+1))
    
    pseudo_resids, resids = estimator.transform(est_quantiles.values, actual.values)
    
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
    axes[1,0].set_xlabel("Predicted")
    axes[1,0].set_ylabel("Actual")
    
    
    a, c = acf(resids, alpha = 0.05, nlags = 50)
    print(np.isnan(resids).sum())
    axes[1,1].bar(np.arange(0,len(a)),a)
    axes[1,1].fill_between(np.arange(0,len(a)), c[:,0] - a, c[:,1] - a, color = 'black', alpha = 0.3)
    
    return resids, axes
    
class quantile_estimator():
    def __init__(self, quantiles = [0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99]):
        self.quantiles = np.array(quantiles[:])
        
    def fit(self, quantiles):
        return
    def forward(self, y):
        return
    def backward(self, u):
        return
    def transform(self, est_quantiles, actuals):
       
        pseudo_resids = np.zeros(len(est_quantiles))
        
        for i in range(len(est_quantiles)):
            self.fit(est_quantiles[i,:])
            pseudo_resids[i] = self.forward(actuals[i])
        
        resids = stats.norm().ppf(pseudo_resids)
        
        return pseudo_resids, resids
    
    def back_transform(self, est_quantiles, resids):
        
        pseudo_resids = stats.norm().cdf(resids)
        
        orig = np.zeros(len(pseudo_resids))
        
        for i in range(len(est_quantiles)):
            self.fit(est_quantiles[i,:])
            orig[i] = self.backward(pseudo_resids[i])
        
        return pseudo_resids, orig
    
class constant_estimator(quantile_estimator):
    
    def fit(self, est_quantiles):
        self.q = est_quantiles[:]
        
    def forward(self, y):
        tmp = y > self.q
        if tmp.any(): 
            return self.quantiles[tmp][-1]
        else:
            return self.quantiles[0]
    def backward(self, u):

        tmp = u == self.q
        if tmp.any(): return self.q[tmp]
        else: return np.nan
     
class piecewise_linear(quantile_estimator):
    
    def __init__(self, quantiles, min_val, max_val):
        
        super().__init__(quantiles)
        
        tmp = np.zeros(len(self.quantiles)+2)
        tmp[1:-1] = self.quantiles
        tmp[0] = 0
        tmp[-1] = 1
        
        self.quantiles = tmp
        
        self.max_val = max_val
        self.min_val = min_val
    
    
    def fit(self, est_quantiles):
        
        self.q_vals = np.zeros(len(est_quantiles)+2)
        self.q_vals[1:-1] = est_quantiles
        
        self.q_vals[-1] = self.max_val
        self.q_vals[0] = self.min_val

        self.diffs = self.q_vals[1:] - self.q_vals[:-1]
        self.coefs = self.quantiles[1:] - self.quantiles[:-1]
    
    def forward(self, y):
        
        conds = (y >= self.q_vals[:-1]) & (y < self.q_vals[1:])
        vals = self.coefs * (y - self.q_vals[:-1]) / self.diffs + self.quantiles[:-1]

        return np.piecewise(y, conds, vals)
    
    def backward(self, u):
        
        conds = (u >= self.quantiles[:-1]) & (u < self.quantiles[1:])
        vals = self.diffs * (u-self.quantiles[:-1]) / self.coefs + self.q_vals[:-1]

        return np.piecewise(u, conds, vals)
    
   