import pathlib
import numpy as np
import pandas as pd

import nabqra

#from pandas import IndexSlice as idx
from collections import defaultdict

#%% load data

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "NABQR"
save_path = PATH / "Data" / "Autocorrelation"

taqr_quantiles = pd.read_pickle(load_path / "taqr_quantiles.pkl")
lstm_basis = pd.read_pickle(load_path / "corrected_ensembles.pkl")
lstm_quantiles = pd.read_pickle(load_path / "lstm_quantiles.pkl")
actuals = pd.read_pickle(load_path / "actuals.pkl")

zones = actuals.columns.get_level_values(0).unique()
date_index = taqr_quantiles.index
# i used the same quantiles for lstm and taqr, change if this is changed
quantiles_str = np.array(taqr_quantiles.columns.get_level_values(1).unique())
quantiles = quantiles_str.astype(np.float64)

zone_limits ={
    "DK1-offshore": (0, 1300),
    "DK1-onshore": (0,3600),
    "DK2-offshore": (0,1100),
    "DK2-onshore": (0, 650)
    }

# there are some quantile crossings
# fix by sorting
# problem seems worse for taqr than for lstm
for zone in zones:
    taqr_quantiles[zone] = nabqra.misc.fix_quantiles(taqr_quantiles[zone], *zone_limits[zone])
    lstm_quantiles[zone] = nabqra.misc.fix_quantiles(lstm_quantiles[zone], *zone_limits[zone])

# ensure everything has the same datapoints
taqr_quantiles = taqr_quantiles.loc[date_index,:]
lstm_basis = lstm_basis.loc[date_index,:]
lstm_quantiles = lstm_quantiles.loc[date_index,:]
actuals = actuals.loc[date_index]

# sometimes we get negative obs
# gives infinities in normal space,
#gives the models problems
actuals[actuals < 0.01] = 0.01

#%% models

global_params = {"n_sim": 200}

models = {
    "Pure Ar": {
        "basis": "actuals",
        "quantile model": None,
        "quantile params": {},
        "correlation model": nabqra.correlation.sarma,
        "correlation params": {"order":(4,0,0), "trend": 'c'}
        },
    "LSTM":{
        "basis": "lstm",
        "quantile model": nabqra.quantiles.spline_model,
        "quantile params": {},
        "correlation model": nabqra.correlation.dummy,
        "correlation params": {}
        },
    "LSTM + SARMA": {
        "basis": "lstm",
        "quantile model": nabqra.quantiles.spline_model,
        "quantile params": {},
        "correlation model": nabqra.correlation.sarma,
        "correlation params": {"order": (1,0,1), "seasonal_order": (1,0,1,24)}
        },
    "NABQR": {
        "basis": "taqr",
        "quantile model": nabqra.quantiles.spline_model,
        "quantile params": {},
        "correlation model": nabqra.correlation.dummy,
        "correlation params": {}
        },
    "NABQR + SARMA": {
        "basis": "taqr",
        "quantile model": nabqra.quantiles.spline_model,
        "quantile params": {},
        "correlation model": nabqra.correlation.sarma,
        "correlation params": {"order": (1,0,1), "seasonal_order": (1,0,1,24)}
        }
    }


#%% make data frames

forecast_res = defaultdict(lambda : dict())
simulation_res = defaultdict(lambda : dict())

for zone, lim in zone_limits.items():
    for model, params in models.items():
        print(f'{zone}, {model}:')
        
        obs = actuals[zone]
        
        match params["basis"]:
            case "actuals":
                estimated_quantiles = None
            case "lstm":
                estimated_quantiles = lstm_quantiles[zone]
            case "taqr":
                estimated_quantiles = taqr_quantiles[zone]
            case wtf_mate:
                print("you did something weird, basis:")
                print(wtf_mate)
                raise ValueError
        
        if params["quantile model"] is not None:
            qm = params["quantile model"](quantiles,
                                          *zone_limits[zone],
                                          **params["quantile params"])
        else:
            qm = None
        
        cm = params["correlation model"](**params["correlation params"],**global_params,)
        
        pipeline = nabqra.pipeline.pipeline(cm, qm)
        
        res = pipeline.run(estimated_quantiles, obs)
        
        forecast_res[zone][model] = res[0]
        simulation_res[zone][model] = res[1]

