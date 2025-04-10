import pathlib
import os
os.chdir(pathlib.Path.cwd().parents[1].resolve())

import numpy as np
import pandas as pd


from pandas import IndexSlice as idx

import Code.quantiles as q_models
import Code.evaluation as evaluation
import Code.pipeline as pipeline
import Code.correlation as corr_models


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
