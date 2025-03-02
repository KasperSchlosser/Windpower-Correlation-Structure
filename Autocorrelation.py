
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import nabqr
import properscoring as ps

import pathlib

import Code.quantiles as qm

from pandas import IndexSlice as idx
from itertools import product

#%%
def get_forecast(modelres, resids, N_step = 24, N_sim = 0, alpha = 0.1):
    # makes an n-step forecast using the fitter model
    #    model should ideally be refit every forecast, but SARMA model are relatively simple, so should be fine 
    # model: (for now) stastmodels SARIMAX model fit result
    # resids: (n,) array of residuals in normal space
    # N: number of data points to forecast
    # Return_sim: number of forecast simulations to make
    
    res = np.zeros((len(resids), 3))
    sim_res = np.zeros((len(resids), N_sim))
    
    #simulate first chunk if needed
    
    if N_sim > 0: 
        sim_res[:N_step, :] = modelres.simulate(nsimulations = N_step, repetitions = N_sim).squeeze()
    
    for k in range(N_step, len(resids) - N_step, N_step):
        # seems to take a bit longer to fit, all data might not be need
        
        min_k = max(k - 10 * N_step,0)
        res2 = modelres.apply(resids[min_k:k])
        
        tmp = res2.get_forecast(N_step)
        res[k:k + N_step,:] = tmp.summary_frame(alpha = alpha).values[:,[0,2,3]] # discard mean_se
        
        if N_sim > 0:
            sim_res[k:k + N_step, :] = res2.simulate(nsimulations = N_step, repetitions = N_sim, anchor = "end").squeeze()
    
    #get last chunk if needed
    last = (len(resids) // N_step) * N_step
    k_last = len(resids) - last
    
    if k_last < N_step:
        res2 = modelres.apply(resids[:last])
        
        tmp = res2.get_forecast(k_last)
        res[last:,:] = tmp.summary_frame(alpha = alpha).values[:,[0,2,3]] # discard mean_se
        if N_sim > 0:
            sim_res[last:, :] = res2.simulate(nsimulations = k_last, repetitions = N_sim, anchor = "end").squeeze()
            
    return res, sim_res

def variogram_score(x, y, p = 0.5, horizon = 24):
    #basically bastians code
    # x: k*n matrix, k = number of simulations, n = number of obersvations
    # y: (n,) array of y
    
    k,n = x.shape
    
    scores = np.zeros((horizon,horizon))
    for t in range(0, len(y) // horizon):
        for i in range(t*horizon, (t+1)*horizon - 1):
            for j in range(i+1 , t + horizon):
                dist =  (j-i)
                Ediff = np.mean(np.abs(x[:,i] - x[:,j])**p)
                Adiff = np.abs(y[i]- y[j])**p
                         
                scores[i,j] += 1 / dist * (Adiff - Ediff) ** 2
    return scores.sum(), scores
    

#%% load data
PATH = pathlib.Path()

data = pd.read_pickle( PATH / "Data" / ("NABQR_results_full.pkl" ))
zones = data.columns.get_level_values(level = 0).unique()
quantiles = data[:]["Quantiles"].columns.unique().values

horizons = [6, 12, 24]
N_sim = 10

# there is some quantile crossings
# i will try to fix by just sorting
# might work?
for zone in zones:
    data[zone, "Quantiles"] = np.sort(data[zone, "Quantiles"].values)


#%%
df_forecast = pd.DataFrame(index = data.index,
                           columns = pd.MultiIndex.from_product([zones, ["nabqr"] + horizons, ("normal", "original"), ("estimate", "lower prediction", "upper prediction")]),
                           dtype = np.float64
)

df_sim = pd.DataFrame(index = data.index,
                           columns = pd.MultiIndex.from_product([zones, ["nabqr"] + horizons, ("normal", "original"), np.arange(N_sim)]),
                           dtype = np.float64
) 

#%% nabqr
df_forecast.loc[:, idx[:,"nabqr","normal", "estimate"]]= stats.norm.ppf(0.5)
df_forecast.loc[:, idx[:,"nabqr","normal", "lower prediction"]] = stats.norm.ppf(0.05)
df_forecast.loc[:, idx[:,"nabqr","normal", "upper prediction"]] = stats.norm.ppf(0.95)

df_sim.loc[:, idx[:, "nabqr", "normal", :]] = stats.norm().rvs((len(data), len(zones) * N_sim))

for zone in zones:
    print(zone)
    
    actuals = data[zone,"Observed"].values.squeeze()
    est_quantiles = data[zone, "Quantiles"].values.squeeze()
    
    quant_est = qm.piecewise_linear_model(quantiles, actuals.min() - 1, actuals.max()*1.1)
    resids, _ = quant_est.transform(est_quantiles, actuals)
    resids = resids.squeeze()
    
    for horizon in horizons:
        print(horizon)
        model = sm.tsa.SARIMAX(resids, order = (1,0,1), seasonal_order=(1,0,1,24))
        res = model.fit()
        
        tmp = get_forecast(res, resids, N_step = horizon, N_sim = N_sim)
        
        df_forecast.loc[:, idx[zone, horizon, "normal", :]] = tmp[0]
        df_sim.loc[:, idx[zone, horizon, 'normal', :]] = tmp[1]
    
    df_forecast.loc[:, idx[zone, :, "original", :]] = quant_est.back_transform(est_quantiles, df_forecast.loc[:, idx[zone,:,"normal",:]].values)[0]
    df_sim.loc[:, idx[zone, :, "original", :]] = quant_est.back_transform(est_quantiles, df_sim.loc[:, idx[zone,:,"normal",:]].values)[0]
    

#%% scores

df_scores = pd.DataFrame(index = ["nabqr"] + list(horizons),
                         columns = pd.MultiIndex.from_product([zones,["MAE", "MSE", "VARS", "CRPS"]]),
                         dtype = float)

for zone in zones:
    actuals = data[zone,"Observed"].values
    predicted = df_forecast.loc[:,idx[zone, :, "original", "estimate"]]

    df_scores.loc[:, idx[zone,"MAE"]] = (predicted - actuals).abs().mean().values
    df_scores.loc[:, idx[zone,"MSE"]] = ((predicted - actuals)**2).mean().values
    
    for p in df_scores.index:
        
        sim = df_sim.loc[:, idx[zone, p, "original", :]]
        
        df_scores.loc[p, idx[zone,"VARS"]] = nabqr.variogram_score_R_multivariate(sim.values.T, actuals.squeeze())[0]
        
        df_scores.loc[p, idx[zone,"CRPS"]] = nabqr.calculate_crps(actuals.squeeze(), sim.values)

#%% plots

for zone in zones:
    fig, ax = plt.subplots(figsize = (14,8), layout = "tight")
    sns.scatterplot(data = data[zone,"Observed"].reset_index(names = "Date"),
                    x = "Date", y = "Observed", color = "black", marker = 'x', ax = ax)
    sns.lineplot(x = df_forecast.index,
                 y = df_forecast[zone,"original", "estimate"],
                 color = 'crimson', ax = ax)
    sns.lineplot(x = data.index,
                 y = data[zone,"Quantiles", "0.50"],
                 color = 'navy', ax = ax)
    ax.fill_between(df_forecast.index,
                    df_forecast[zone,"original", "0.05"],
                    df_forecast[zone,"original", "0.95"],
                    color = "crimson", alpha = 0.3)
    ax.fill_between(data.index,
                    data[zone,"Quantiles", "0.05"],
                    data[zone,"Quantiles", "0.95"],
                    color = "navy", alpha = 0.3)
    ax.vlines(data.index[23::24], np.nanmin(df_forecast[zone].values), np.nanmax(df_forecast[zone].values), color = 'black', linestyle = '--')
    ax.set_title(zone)
    
    fig.savefig(PATH / "Figures" / "Correlation Structure" / (zone + "_forecast.png"))

for i, zone in enumerate(zones):
    
    fig, ax = plt.subplots(figsize = (14,8), layout = "tight")
    
    sns.scatterplot(x = np.squeeze(df_forecast.loc[:,idx[zone,'original', 'estimate']].values),
                    y = np.squeeze(data[zone,"Observed"].values),
                    color = 'crimson',
                    ax = ax)
    sns.scatterplot(x = np.squeeze(data.loc[:,idx[zone,"Quantiles","0.50"]].values),
                    y = np.squeeze(data[zone,"Observed"].values),
                    color = 'navy', alpha = 0.5)
    ax.axline((0,0), slope = 1, color = 'black')
    ax.set_title(zone)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    fig.savefig(PATH / "Figures" / "Correlation Structure" / (zone + "actual_predicted.png"))
    