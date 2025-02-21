# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:49:21 2025

@author: KPFS
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm


#%% quantile models
# models estimating cdfs from the nabqr quantiles
# models fits a function to quantiles model(q) -> F(x)
# foward takes and observation x and gives resulting quantile:
#   model.forward(y): F(y) = u
# Backwards takes from cdf-space back to original space
# model.backward(u): F^-1(u) = x

class quantile_model():
    def __init__(self, quantiles = (0.5,0.3,0.5,0.7,0.95), dist = stats.norm()):
        self.quantiles = np.array(quantiles)
        self.dist = dist
        
    def fit(self, est_quantiles):
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

class constant_model(quantile_model):
    
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
     
class piecewise_linear_model(quantile_model):
    
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

#%% misc functions

def evaluate_pseudoresids(pseudo_resids, index = None, save_path = None, name = None, figsize = (14,8), close_figs = False):
    if index is None:
        index = list(range(len(pseudo_resids)))
    
    if name is None:
        Name = "psudo_residual"
    
    dist = stats.norm()
    resids = dist.ppf(pseudo_resids)
    
    theo_quantiles = dist.ppf(np.arange(1,len(pseudo_resids)+1) / (len(pseudo_resids)+1))
    
    figs = [plt.figure(figsize = figsize, layout = "tight") for _ in range(5)]
    
    # Scatter plot
    ax = figs[0].subplots()
    sns.scatterplot(x = index,  y = resids, ax = ax)
    
    lims = (dist.ppf(0.05 / (2*len(pseudo_resids))), dist.ppf(1 - 0.05 / (2*len(pseudo_resids))))
    ax.hlines(0, index[0], index[-1], color = 'black')
    ax.hlines((-2,2), index[0], index[-1], color = 'navy', linestyle = 'dashed')
    ax.hlines(lims, index[0], index[-1], color = "crimson")
    
    if save_path is not None:
        figs[0].savefig( save_path / (Name + "_scatter.png" ))
        
    # Uniform histogram
    ax = figs[1].subplots()
    sns.histplot(x = resids, stat = "density", ax = ax)
    
    if save_path is not None:
        figs[1].savefig( save_path / (Name + "_cdfdist.png" ))
    
    
    #normal histogram
    ax = figs[2].subplots()
    
    sns.histplot(x = resids, stat = "density", ax = ax)
    sns.lineplot(x = theo_quantiles, y = dist.pdf(theo_quantiles), ax = ax, color = 'black')
    if save_path is not None:
        figs[2].savefig( save_path / (Name + "_normaldist.png" ))
    
    # propplot
    ax = figs[3].subplots()
    
    sns.scatterplot(x = theo_quantiles, y = sorted(resids), ax = ax)
    ax.axline((theo_quantiles[0], theo_quantiles[0]), (theo_quantiles[-1], theo_quantiles[-1]))
    ax.set_xlabel("Theoretical Quantile")
    ax.set_ylabel("Observed Quantile")
    
    if save_path is not None:
        figs[3].savefig( save_path / (Name + "_probplot.png" ))
    
    # (pacf) plot
    axs = figs[4].subplots(1,2)
    
    acf_vals, conf_acf = sm.tsa.acf(resids, alpha = 0.05)
    pacf_vals, conf_pacf = sm.tsa.pacf(resids, alpha = 0.05)
    
    sns.barplot(x = np.arange(len(acf_vals)), y = acf_vals, ax = axs[0])
    axs[0].fill_between(
        np.arange(len(acf_vals)),
        conf_acf[:,0] - acf_vals,
        conf_acf[:,1] - acf_vals,
        color = 'black',
        alpha = 0.3
    )
    axs[0].set_title("ACF")
    
    sns.barplot(x = np.arange(len(pacf_vals)), y = pacf_vals, ax = axs[1])
    axs[1].fill_between(
        np.arange(len(pacf_vals)),
        conf_pacf[:,0] - pacf_vals,
        conf_pacf[:,1] - pacf_vals,
        color = 'black',
        alpha = 0.3
    )
    axs[1].set_title("PACF")
    
    if save_path is not None:
        figs[4].savefig( save_path / (Name + "_autocorrelation.png" ))
        
    if close_figs:
        for fig in figs:
            plt.close(fig)
        return
    
    return figs
