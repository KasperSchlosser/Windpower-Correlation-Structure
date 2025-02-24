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
import statsmodels.api as sm

from statsmodels.tsa.api import acf

from functions import evaluate_pseudoresids, piecewise_linear_model

#%% define stuff

#i think these are the quantiles Bastian used
#quantiles = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
#full centiles
quantiles = np.arange(0.01,1,0.01);
train_proportion = 0.8


#%% load data

# There is a small problem with the data
# the observations has no timezone, while the ensembles have
# i belive the timezone have just been striped from the observation data
# the result is there are 3 nan values in the observation data
# additionally just stripping the timezone from the ensemble results in two rows with duplicated index.

#to match data we have to strip timezones -> drop one of the duplicated dates -> drop nans from observations

ensembles = pd.read_pickle("data/ensembles_input_all/DK2_offshore_wind_power_oct24.pkl")
#strip timezone from ensembles
ensembles.index = ensembles.index.to_series().apply(lambda x: x.tz_localize(None)).values
ensembles = ensembles[~ensembles.index.duplicated()]
obs = pd.read_pickle("data/observations/actuals_DK2_offshore_wind_power_oct24.pkl")
obs = obs.dropna()


#%% fit nabqr

#use nabqr to estimate quantiles

corrected, estimated_quantiles, actual, beta, orig  = nq.pipeline(ensembles,
                                                                  obs.values,
                                                                  training_size = train_proportion,
                                                                  epochs = 200,
                                                                  quantiles_taqr = quantiles)
estimated_quantiles.columns = [f'{x:.02f}' for x in quantiles]

#%%
piecewise_estimator = piecewise_linear([float(x) for x in estimated_quantiles.columns], actual.min()-10, actual.max()+10)
pseudo, resids = piecewise_estimator.transform(estimated_quantiles.values, actual.values)

#%%
evaluate_pseudoresids(actual, estimated_quantiles, piecewise_estimator)


#%% 


# from manual testing best model seem to be a arma(1,1) model
# seaonal component slightly lower AIC and BIC.
# however i belive this comes from the fewer data points used in calculating -ll

ar_mod = sm.tsa.SARIMAX(resids, order = (1,0,1),seasonal_order=(1,0,1,24))
res = ar_mod.fit()


#%%

#evaluate ar model
#maybe be done more in detail

print(res.bic)
fig, axes = plt.subplots(2, figsize = (14,16))
_ = sm.graphics.tsa.plot_acf(res.resid, ax =  axes[0])
_ = sm.graphics.tsa.plot_pacf(res.resid, ax = axes[1])

#%%

pred_res = res.get_prediction()

orig_corrected = pd.DataFrame( {"median": piecewise_estimator.back_transform(estimated_quantiles.values,pred_res.predicted_mean)[1]})

orig_corrected.index = estimated_quantiles.index


for a in [0.6, 0.1, 0.02]:
    
    tmp = pred_res.conf_int(alpha = a)
    print(tmp)
    
    orig_corrected[str(a/2)] = piecewise_estimator.back_transform(estimated_quantiles.values, tmp[:,0])[1]
    orig_corrected[str(1-a/2)] = piecewise_estimator.back_transform(estimated_quantiles.values, tmp[:,1])[1]

#%% plot forcasts and actuales

fig = plt.figure(figsize = (14,8))
X = estimated_quantiles.index
#nabqr
plt.plot(X, estimated_quantiles["0.50"], color = 'blue')
#plt.plot(X, estimated_quantiles[["0.30","0.70"]], color = 'blue', linestyle = ':')
plt.plot(X, estimated_quantiles[["0.05","0.95"]], color = 'blue', linestyle = '--')
#plt.plot(X, estimated_quantiles[["0.01","0.99"]], color = 'blue', linestyle = '-.')

#corrected
plt.plot(X, orig_corrected["median"], color = 'red')
#plt.plot(X, orig_corrected[["0.3","0.7"]], color = 'red', linestyle = ':')
plt.plot(X, orig_corrected[["0.05","0.95"]], color = 'red', linestyle = '--')
#plt.plot(X, orig_corrected[["0.01","0.99"]], color = 'red', linestyle = '-.')

plt.scatter(X, actual, color = 'black', marker = 'x')

ax = plt.gca()
ax.set_xlim([np.datetime64("2024-08-01"),np.datetime64("2024-08-10")])
fig.show()



#%%

plt.figure(figsize = (14,8))

plt.plot(estimated_quantiles.index, estimated_quantiles["0.50"], color = 'black')
plt.fill_between(estimated_quantiles.index, estimated_quantiles["0.05"],estimated_quantiles["0.95"], color = 'blue', alpha = 0.3)
plt.scatter(estimated_quantiles.index, orig_corrected["median"], c = 'b', marker = '*')
plt.scatter(estimated_quantiles.index, actual, c = 'k', marker = 'x')
plt.xlim([np.datetime64("2024-08-01"),np.datetime64("2024-08-31")])

plt.figure(figsize = (14,8))
tmp = pd.DataFrame({"Actual": actual, "uncorrected": estimated_quantiles["0.50"], "corrected": orig_corrected["median"]})
sns.pairplot(tmp)
print(tmp.corr())


#%% plot tails problems
# this will be better fixed with a continous distribution


