
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


#%% load data
PATH = pathlib.Path()

data = pd.read_pickle( PATH / "Data" / ("NABQR_results_full.pkl" ))
zones = data.columns.get_level_values(level = 0).unique()
quantiles = data[:]["Quantiles"].columns.unique().values

# there is some quantile crossings
# i will try to fix by just sorting
# might work?
for zone in zones:
    data[zone, "Quantiles"] = np.sort(data[zone, "Quantiles"].values)

#make transformed data

data_resids = pd.DataFrame(index = data.index)
for zone in zones:
    df_zone = data[zone]
    
    quant_est = qm.piecewise_linear_model(quantiles, data[zone].values.min(), data[zone].values.max())
    _, resids = quant_est.transform(df_zone["Quantiles"].values, df_zone["Observed"].values.squeeze())
    
    data_resids[zone] = resids

#%% individual SARMA models.

df_individual = pd.DataFrame(index = data.index,
                             columns = pd.MultiIndex.from_product([zones, ("normal", "original"),("estimate","0.05", "0.95")]),
                             dtype = np.float64
)

df_forecast = pd.DataFrame(index = data.index,
                           columns = pd.MultiIndex.from_product([zones, ("normal", "original"),("estimate","0.05", "0.95")]),
                           dtype = np.float64
)

N_forecast = 24

for zone in zones:
    print(zone)
    df_zone = data[zone]
    
    quant_est = qm.piecewise_linear_model(quantiles, data[zone,"Observed"].values.min()-1, data[zone,"Observed"].values.max()*1.1)
    _, resids = quant_est.transform(df_zone["Quantiles"].values, df_zone["Observed"].values.squeeze())
    
    
    model = sm.tsa.SARIMAX(resids, order = (1,0,1), seasonal_order=(1,0,1,24))
    res = model.fit()
    
    #one step
    df_individual[zone,"normal","estimate"] = res.get_prediction().predicted_mean
    conf_int = res.get_prediction().conf_int(alpha = 0.1)
    df_individual[zone,"normal", "0.05"] = conf_int[:,0]
    df_individual[zone,"normal", "0.95"] = conf_int[:,1]
    
    #this needs to be fixed in the quantile estimator
    for est in df_individual.columns.get_level_values(2).unique():
        tmp = quant_est.back_transform(df_zone["Quantiles"].values, df_individual[zone,"normal", est].values.squeeze())[1]
        df_individual[zone,"original", est] = tmp
    
    #forecast
    for i in np.arange(0,len(resids)-N_forecast, N_forecast):
        ind = df_forecast.index[i:i+N_forecast]
        
        res2 = res.apply(resids[:i])
        df_forecast.loc[ind, idx[zone,"normal","estimate"]] = res2.get_forecast(N_forecast).predicted_mean
        df_forecast.loc[ind, idx[zone,"normal",["0.05", "0.95"]]] = res2.get_forecast(N_forecast).conf_int(0.05)
        
    # get last chunck
    i = (len(resids) // N_forecast)*N_forecast + 1
    ind = df_forecast.index[i:]
    
    res2 = res.apply(resids[:i])
    df_forecast.loc[ind, idx[zone,"normal","estimate"]] = res2.get_forecast(len(resids) - i).predicted_mean
    df_forecast.loc[ind, idx[zone,"normal",["0.05", "0.95"]]] = res2.get_forecast(len(resids) - i).conf_int(0.05)
    
    for est in df_forecast.columns.get_level_values(2).unique():
        tmp = quant_est.back_transform(df_zone["Quantiles"].values, df_forecast[zone,"normal", est].values.squeeze())[1]
        df_forecast[zone,"original", est] = tmp
    

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
    
#%% calculate scores


df_scores = pd.DataFrame(index = ["NABQR", "prediction", "forecast"],
                         columns = pd.MultiIndex.from_product([zones,["MAE", "MSE", "-ll", "VARS", "CRPS"]]),
                         dtype = float)

actual = data.xs("Observed", level = 2, axis = 1).values

df_scores.loc["NABQR", idx[:, "MAE"]] = np.mean(np.abs((actual - data.xs("0.50", level = 2, axis = 1).values)), axis = 0)
df_scores.loc["NABQR", idx[:, "MSE"]] = np.mean((actual - data.xs("0.50", level = 2, axis = 1).values)**2, axis = 0)

df_scores.loc["prediction", idx[:, "MAE"]] = np.mean(np.abs((actual - df_individual.xs("original", level = 1, axis = 1).xs("estimate", level = 1, axis = 1).values)), axis = 0)
df_scores.loc["prediction", idx[:, "MSE"]] = np.mean((actual - df_individual.xs("original", level = 1, axis = 1).xs("estimate", level = 1, axis = 1).values)**2, axis = 0)

df_scores.loc["forecast", idx[:, "MAE"]] = np.mean(np.abs((actual - df_forecast.xs("original", level = 1, axis = 1).xs("estimate", level = 1, axis = 1).values)), axis = 0)
df_scores.loc["forecast", idx[:, "MSE"]] = np.mean((actual - df_forecast.xs("original", level = 1, axis = 1).xs("estimate", level = 1, axis = 1).values)**2, axis = 0)


# -ll
for zone in zones:
    # get residuals
    quant_est = qm.piecewise_linear_model(quantiles, data[zone,"Observed"].values.min()-1, data[zone,"Observed"].values.max()*1.1)
    _, resids = quant_est.transform(data[zone,"Quantiles"].values, data[zone,"Observed"].values.squeeze())
    
    # -ll for nabqr is just the actuals
    # this score is a bit funky in th
    
    df_scores.loc["NABQR", idx[zone, "-ll"]] = -stats.norm().logpdf(resids).sum()
    
    # i can get the std of the sarma model from the conf int
    
    sd =  (df_forecast[zone, "normal", "estimate"] - df_forecast[zone, "normal", "0.05"]) / stats.norm().ppf(0.95)
    ll = stats.norm().logpdf((resids - df_forecast[zone, "normal", "estimate"]) / sd)
    df_scores.loc["forecast", idx[zone, "-ll"]] = -np.ma.masked_invalid(ll).sum()
    
    sd =  (df_individual[zone, "normal", "estimate"] - df_individual[zone, "normal", "0.05"]) / stats.norm().ppf(0.95)
    ll = stats.norm().logpdf((resids - df_individual[zone, "normal", "estimate"]) / sd)
    df_scores.loc["prediction", idx[zone, "-ll"]] = -np.ma.masked_invalid(ll).sum()
    

# vars
# i use nabqr implementation
for i, zone in enumerate(zones):
    
    y = actual[:,i]
    x_nabqr = data[zone, "Quantiles", "0.50"].values.reshape((1,-1))
    x_prediction = df_individual[zone, "original", "estimate"].values.reshape((1,-1))
    x_forecast = df_forecast[zone, "original", "estimate"].values.reshape((1,-1))
    
    df_scores.loc["NABQR", idx[zone, "VARS"]] = nabqr.variogram_score_R_multivariate(x_nabqr, y, t1 = N_forecast, t2 = 2*N_forecast )[0]
    df_scores.loc["prediction", idx[zone, "VARS"]] = nabqr.variogram_score_R_multivariate(x_prediction, y, t1 = N_forecast, t2 = 2*N_forecast )[0]
    df_scores.loc["forecast", idx[zone, "VARS"]] = nabqr.variogram_score_R_multivariate(x_forecast, y, t1 = N_forecast, t2 = 2*N_forecast )[0]

#%%
# CRPS
# this seems harder for nabqr quantiles
# not done for now
for zone in zones:
    y = data[zone,"Observed"].values.squeeze()
    x_min = y.min()-1
    x_max = y.max()*1.1
    dist = stats.norm()
    
    quant_est = qm.piecewise_linear_model(quantiles, data[zone,"Observed"].values.min()-1, data[zone,"Observed"].values.max()*1.1)
    
    sd_forecast =  ((df_forecast[zone, "normal", "estimate"] - df_forecast[zone, "normal", "0.05"]) / stats.norm().ppf(0.95)).values
    mu_forecast = df_forecast[zone, "normal", "estimate"].values
    
    sd_prediction =  ((df_individual[zone, "normal", "estimate"] - df_individual[zone, "normal", "0.05"]) / stats.norm().ppf(0.95)).values
    mu_prediction = df_individual[zone, "normal", "estimate"].values
    
    crps_nabqr = 0
    crps_pred = 0
    crps_fore = 0
    for i in range(1,len(y)): #skip first, does not work for sarma models
        
        quant_est.fit(data[zone,"Quantiles"].iloc[i])
        
        cdf_nabqr = lambda x: quant_est.forward(x)
        cdf_prediction = lambda x: dist.cdf((dist.ppf(quant_est.forward(x)) - mu_prediction[i]) / sd_prediction[i])
        cdf_forecast = lambda x: dist.cdf((dist.ppf(quant_est.forward(x)) - mu_forecast[i]) / sd_forecast[i])
        
        crps_nabqr += ps.crps_quadrature(y[i], cdf_nabqr, x_min, x_max)
        crps_pred += ps.crps_quadrature(y[i], cdf_prediction, x_min, x_max )
        crps_fore += ps.crps_quadrature(y[i], cdf_forecast, x_min, x_max )
        
    df_scores.loc["NABQR", idx[zone, "CRPS"]] = crps_nabqr / len(y)-1
    df_scores.loc["prediction", idx[zone, "CRPS"]] = crps_pred / len(y)-1
    df_scores.loc["forecast", idx[zone, "CRPS"]] = crps_fore / len(y)-1
        
    


#%%
# qss
df_qss =  pd.DataFrame(index = quantiles, columns = pd.MultiIndex.from_product([zones, ["NABQR", "prediction", "forecast"]]), dtype = float)
for i, zone in enumerate(zones):
    print(zone)
    
    quant_est = qm.piecewise_linear_model(quantiles, data[zone,"Observed"].values.min()-1, data[zone,"Observed"].values.max()*1.1)
    
    y = actual[:,i]
    #this is the sorted quantiles
    #might be worth to compare to non sorted
    x_nabqr = data.loc[:,idx[zone, "Quantiles", :]].values
    
    #need to calculate the quantiles for forcast and onestep-prediction
    Q = stats.norm().ppf(quantiles.astype(float))
    #need to transform back
    #gonna take some compute time
    
    #forecast
    sd =  (df_forecast[zone, "normal", "estimate"] - df_forecast[zone, "normal", "0.05"]) / stats.norm().ppf(0.95)
    tmp = np.outer(sd, Q) + df_forecast[zone, "normal", "estimate"].values[:,np.newaxis]
    x_forecast = np.zeros_like(tmp)
    
    print("forecast")
    for k in range(tmp.shape[1]):
        x_forecast[:,k] = quant_est.back_transform(df_zone["Quantiles"].values, tmp[:,k])[1]
    
    #onestep prediction
    sd =  (df_individual[zone, "normal", "estimate"] - df_individual[zone, "normal", "0.05"]) / stats.norm().ppf(0.95)
    tmp = np.outer(sd, Q) + df_individual[zone, "normal", "estimate"].values[:,np.newaxis]
    x_prediction = np.zeros_like(tmp)
    
    print("prediction")
    for k in range(tmp.shape[1]):
        x_prediction[:,k] = quant_est.back_transform(df_zone["Quantiles"].values, tmp[:,k])[1]
    
    df_qss.loc[:, idx[zone, "NABQR"]] = nabqr.functions.multi_quantile_skill_score(y, x_nabqr, quantiles.astype(float))
    df_qss.loc[:, idx[zone, "prediction"]] = nabqr.functions.multi_quantile_skill_score(y, x_prediction, quantiles.astype(float))
    df_qss.loc[:, idx[zone, "forecast"]] = nabqr.functions.multi_quantile_skill_score(y, x_forecast, quantiles.astype(float))
    

#%% combined sarma model
"""
df_combined = pd.DataFrame(index = data.index,
                           columns = pd.MultiIndex.from_product([zones, ("normal", "original"),("estimate","0.05", "0.95")]),
                           dtype = np.float64
)

tmp = pd.DataFrame( index = data.index, columns = zones, dtype = float)
for zone in zones:
    quant_est = qm.piecewise_linear_model(quantiles, data[zone,"Observed"].values.min()-1, data[zone,"Observed"].values.max()*1.1)
    _, resids = quant_est.transform(data[zone,"Quantiles"].values, data[zone,"Observed"].values.squeeze())
    
    tmp.loc[:, zone] = resids.astype(float)

mv_model = sm.tsa.VARMAX(tmp.values, order = (1,0))
res = mv_model.fit()

df_combined.loc[:,idx[:,"normal","estimate"]] = res.get_prediction().predicted_mean
df_combined.loc[:,idx[:,"normal",["0.05","0.95"]]] = res.get_prediction().conf_int(0.1)

for zone in zones:
    for est in df_combined.columns.get_level_values(2).unique():
        tmp = quant_est.back_transform(data[zone,"Quantiles"].values, df_combined[zone,"normal", est].values.squeeze())[1]
        df_combined[zone,"original", est] = tmp


fig, axes = plt.subplots(2,2,figsize = (14,8), layout = "tight")
axes = axes.flatten()

for i, zone in enumerate(zones):
    
    axes[i].scatter(df_combined.loc[:,idx[zone,'original', 'estimate']], data[zone,"Observed"].values, color = 'crimson')
    axes[i].scatter(data.loc[:,idx[zone,"Quantiles","0.50"]], data[zone,"Observed"].values, color = 'navy', alpha = 0.3)
    axes[i].axline((0,0), slope = 1, color = 'black')
"""
