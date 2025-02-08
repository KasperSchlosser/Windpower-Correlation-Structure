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
corrected, estimated_quantiles, actual, beta, orig  = nq.pipeline(ensembles, obs, epochs = 20, quantiles_taqr = quantiles)
estimated_quantiles.columns = quantiles
#%%

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

#%%

def evaluate_estimator(actual, est_quantiles, estimator):
    X = actual.index
    
    estimator.fit(quantiles)

class quantile_estimator():
    def __init__(self, quantiles = [0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99]):
        self.quantiles = quantiles[:]
        
    def fit(self, quantiles):
        return
    def estimate(self, y):
        return

class constant_estimator(quantile_estimator):
    def fit(self, est_quantiles):
        self.q = est_quantiles[:]
    def estimate(self, y):
        tmp = x > estimated_quantiles.iloc[1,:]
        if tmp.any(): 
            return self.quantiles[tmp][-1]
        else:
            return self.quantiles[0]
        
#%% 
fig = plt.figure()

stats.ecdf(q).cdf.plot()

ax = plt.gca()
ax.plot(np.linspace(0,1,100),np.linspace(0,1,100))
