# This script runs NABQR on the data thens saves.
# should be run sparingly as fitting the lstm take quite a while

import numpy as np
import pandas as pd
import nabqr

from pathlib import Path


quantiles = np.arange(0.01, 1, 0.01)
quantiles_str = [f'{x:.02f}' for x in quantiles]
pipeline_args = {
    "training_size": 0.8,
    "epochs": 120,
    "quantiles_taqr": quantiles 
    }

PATH = Path.cwd()
load_path = PATH.parents[1] / "Data" / "Data"
save_path = PATH.parents[1] / "Data" / "NABQR"

data = pd.read_pickle(load_path / "Cleaned Data.pkl")

zones = data.columns.get_level_values(0).unique()

#%% nabqr


corrected_ensembles = {}
estimated_quantiles = {}
actuals = {}
beta_parameters = {}
original_ensembles = {}
for zone in zones:
    
        c_ens, est_q, act, beta, orig  = nabqr.pipeline(
            data[zone,"Ensembles"],
            data[zone,"Observed"].values.squeeze(),
            name = zone,
            **pipeline_args
        )
        c_names = ["Corrected Ensemble " + str(x+1) for x in c_ens.columns]
        c_ens.columns = c_names
        est_q.columns = quantiles_str
        act.name = "Value"
        beta = pd.concat([pd.DataFrame(ar, columns = c_ens.columns) for ar in beta], axis = 1, keys = quantiles_str)
        orig = orig.loc[c_ens.index,:]

        corrected_ensembles[zone] = c_ens
        estimated_quantiles[zone] = est_q
        actuals[zone] = act
        beta_parameters[zone] = beta
        original_ensembles[zone] = orig
        

        
corrected_ensembles = pd.concat(corrected_ensembles.values(), axis = 1, keys = zones)
estimated_quantiles = pd.concat(estimated_quantiles.values(), axis = 1, keys = zones)
actuals = pd.concat(actuals.values(), axis = 1, keys = zones)
beta_parameters = pd.concat(beta_parameters.values(), axis = 1, keys = zones)
original_ensembles = pd.concat(original_ensembles.values(), axis = 1, keys = zones)


#%%
corrected_ensembles.to_csv(save_path / "corrected_ensembles.csv")
corrected_ensembles.to_pickle(save_path / "corrected_ensembles.pkl")

estimated_quantiles.to_csv(save_path / "estimated_quantiles.csv")
estimated_quantiles.to_pickle(save_path / "estimated_quantiles.pkl")

actuals.to_csv(save_path / "actuals.csv")
actuals.to_pickle(save_path / "actuals.pkl")

beta_parameters.to_csv(save_path / "beta_parameters.csv")
beta_parameters.to_pickle(save_path / "beta_parameters.pkl")

original_ensembles.to_csv(save_path / "original_ensembles.csv")
original_ensembles.to_pickle(save_path / "original_ensembles.pkl")

