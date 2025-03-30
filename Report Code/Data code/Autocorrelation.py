import pathlib
import os
os.chdir(pathlib.Path.cwd().parents[1].resolve())

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

from pandas import IndexSlice as idx

import Code.quantiles as q_models
import Code.evaluation as evaluation
import Code.pipeline as pipeline
import Code.correlation as corr_models

def get_forecast(modelres, resids, N_step = 24, N_sim = 0, alpha = 0.1, burn_in = 240):
    # makes an n-step forecast using the fitter model
    #    model should ideally be refit every forecast, but SARMA model are relatively simple, so should be fine 
    # model: (for now) stastmodels SARIMAX model fit result
    # resids: (n,) array of residuals in normal space
    # N: number of data points to forecast
    # Return_sim: number of forecast simulations to make
    
    res = np.zeros((len(resids), 3))
    sim_res = np.zeros((len(resids), N_sim))
    
    #simulate first chunk if needed
    
    #if N_sim > 0: 
    #    sim_res[:N_step, :] = modelres.simulate(nsimulations = N_step, repetitions = N_sim).squeeze()
    
    for k in range(burn_in, len(resids) - N_step, N_step):
        # seems to take a bit longer to fit, all data might not be need
        
        #min_k = max(k - 10 * N_step,0)
        res2 = modelres.apply(resids[:k], refit = True, copy_initialization = True)
        
        tmp = res2.get_forecast(N_step)
        res[k:k + N_step,:] = tmp.summary_frame(alpha = alpha).values[:,[0,2,3]] # discard mean_se
        
        if N_sim > 0:
            sim_res[k:k + N_step, :] = res2.simulate(nsimulations = N_step, repetitions = N_sim, anchor = "end").squeeze()
    
    #get last chunk if needed
    last = (len(resids) // N_step) * N_step
    k_last = len(resids) - last
    
    if k_last < N_step:
        res2 = modelres.apply(resids[:last], refit = True)
        
        tmp = res2.get_forecast(k_last)
        res[last:,:] = tmp.summary_frame(alpha = alpha).values[:,[0,2,3]] # discard mean_se
        if N_sim > 0:
            sim_res[last:, :] = res2.simulate(nsimulations = k_last, repetitions = N_sim, anchor = "end").squeeze()
            
    return res, sim_res



def correction(actuals, est_quantiles, quant_est, model_type = "SARMA", N_step = 24, N_sim = 100, eval_resids = None, **kwargs):
    
    resids, _ = quant_est.transform(est_quantiles, actuals)
    resids = resids.squeeze()
    
    if model_type == "SARMA":
        model = sm.tsa.SARIMAX(resids, order = (1,0,1), seasonal_order=(1,0,1,24))
    elif model_type == "ARMA":
        model = sm.tsa.SARIMAX(resids, order = (1,0,1))
    model_res = model.fit()
    
    normal_space = get_forecast(model_res, resids, N_step = N_step, N_sim = N_sim, **kwargs)
    
    orignal_forecast, cdf_forecast = quant_est.back_transform(est_quantiles, normal_space[0])
    original_sim, cdf_sim = quant_est.back_transform(est_quantiles, normal_space[1])
    
    if eval_resids is not None:
        save_path = PATH / "Figures" / "Correlation Structure" / "Residuals"
        tmp = model_res.summary().tables[1].as_csv() 
        with open(save_path / (eval_resids + "_" + model_type + "_coefs.csv"), 'w') as f:
            f.write(tmp)
                
        pr = stats.norm().cdf(model_res.resid / np.sqrt(model_res.mse))
        evaluation.evaluate_pseudoresids(pr[1:],
                                   save_path = save_path,
                                   name = eval_resids + "_after",
                                   close_figs = True
                                   )
    
    return normal_space, (cdf_forecast, cdf_sim), (orignal_forecast, original_sim)

#%% load data

PATH = pathlib.Path()
load_path = PATH / "Data" / "NABQR"
save_path = PATH / "Data" / "Autocorrelation"

N_sim = 10

estimated_quantiles = pd.read_pickle(load_path / "estimated_quantiles.pkl")
actuals = pd.read_pickle(load_path / "actuals.pkl")

zones = actuals.columns.get_level_values(0).unique()
quantiles_str = [x for x in estimated_quantiles.columns.get_level_values(1).unique().values]
quantiles = [float(x) for x in quantiles_str]
date_index = actuals.index


# there is some quantile crossings
# i will try to fix by just sorting
# might work?
for zone in zones:
    estimated_quantiles[zone] = np.sort(estimated_quantiles[zone].values)


#%% parameters for models
#max values for each zone
# format: (min, max)
zone_limits ={
    "DK1-offshore": (0, 1250),
    "DK1-onshore": (0,3575),
    "DK2-offshore": (0,1000),
    "DK2-onshore": (0, 640)
    }

qm_params = {zone: (quantiles, *zone_limits[zone]) for zone in zones}

untransformed_sarma_params = {zone: {"order": (1,0,1), "trend":"c", "n_sim": N_sim} for zone in zones}
sarma_params = {zone: {"order": (1,0,1), "seasonal_order": (1,0,1,24),"n_sim": N_sim} for zone in zones}
nabqr_params = {zone: {"n_sim": N_sim} for zone in zones}


#%% make models

models = {
    "NABQR" : {},
    "Untransformed SARMA": {},
    "SARMA" : {}
    }

for zone in zones:
    models["NABQR"][zone] = pipeline.pipeline(
        corr_models.correlation_nabqr(**nabqr_params[zone]),
        q_models.piecewise_linear_model(*qm_params[zone])
    )
    
    models["Untransformed SARMA"][zone] = pipeline.pipeline(
        corr_models.correlation_sarma(**untransformed_sarma_params[zone]),
    )
    
    models["SARMA"][zone] =  pipeline.pipeline(
        corr_models.correlation_sarma(**sarma_params[zone]),
        q_models.piecewise_linear_model(*qm_params[zone])
    )


#%% make data frames

forecast_col = pd.MultiIndex.from_product([
    list(models.keys()),
    zones,
    ("Original", "Cdf", "Normal"),
    ("Observation", "Estimate", "Lower prediction", "Upper prediction")
])
sim_col = pd.MultiIndex.from_product([
    list(models.keys()),
    zones,
    ("Original", "Cdf", "Normal"),
    ("Simulation " + str(x+1) for x in range(N_sim))
])

df_forecast = pd.DataFrame(index = date_index, columns = forecast_col, dtype = np.float64)
df_sim = pd.DataFrame(index = date_index, columns = sim_col, dtype = np.float64)

#%%

for model in models.keys():
    for zone in models[model]:
        
        print(model, zone)
        est_q = estimated_quantiles[zone]
        act = actuals[zone]
        
        tmp = models[model][zone].run(est_q, act)
        df_forecast.loc[:, idx[model,zone, :]] = tmp[0].values
        df_sim.loc[:, idx[model,zone, :]] = tmp[1].values
        

#%% save data

df_forecast.to_csv(save_path / "forecasts.csv")
df_forecast.to_pickle(save_path / "forecasts.pkl")

df_sim.to_csv(save_path / "simulations.csv")
df_sim.to_pickle(save_path / "simulations.pkl")
        

#%% scores
#needs to be made in the forecast evaluation script
df_scores = pd.DataFrame(index = ["nabqr"] + list(models),
                         columns = pd.MultiIndex.from_product([zones,["MAE", "MSE", "VARS", "CRPS"]]),
                         dtype = float)

for zone in zones:
    print(zone)
    #weights = evaluation.variogram_weight(df_sim.loc[:, idx[zone, "nabqr", "original", :]].values)
    
    for p in df_scores.index:
        print(p)
        act = actuals[zone].values.squeeze()[burn_in:]
        predicted = df_forecast.loc[:,idx[zone, p, "original", "estimate"]].values[burn_in:]
        sim = df_sim.loc[:, idx[zone, p, "original", :]].values[burn_in:, :]
        
        #scores = evaluation.calc_scores(actuals, predicted, sim.values, VARS_kwargs = {"weights": weights})
        scores = evaluation.calc_scores(act, predicted, sim)
        
        df_scores.loc[p, idx[zone, :]] = scores
