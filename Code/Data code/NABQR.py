# This script runs NABQR on the data thens saves.
# should be run sparingly as fitting the lstm take quite a while

import numpy as np
import pandas as pd
import nabqr

from pathlib import Path

pipeline_args = {
    "training_size": 0.64, #gives about a year of testing data
    "epochs": 30,
    "quantiles_taqr": np.concat([[0.01], np.arange(0.05,1,0.05), [0.99]]),
    "quantiles_lstm": np.concat([[0.01], np.arange(0.05,1,0.05), [0.99]]),
    }
pipeline_args["names_taqr"] = [f'{x:.2f}' for x in pipeline_args["quantiles_taqr"]]
pipeline_args["names_lstm"] = [f'{x:.2f}' for x in pipeline_args["quantiles_lstm"]]

PATH = Path.cwd()
load_path = PATH.parents[1] / "Data" 
save_path = PATH.parents[1] / "Data" / "NABQR"

data = pd.read_pickle(load_path / "Data" / "Cleaned Data.pkl")

zones = data.columns.get_level_values(0).unique()

#%% nabqr

corrected_ensembles = {}
taqr_quantiles = {}
actuals = {}
beta_parameters = {}
lstm_quantiles = {}
for zone in zones:
        
        # i dont use train data for now
        # really hard to get nabqr to spit that out
        _, test  = nabqr.pipeline(
            data[zone,"Ensembles"],
            data[zone,"Observed"],
            #save_name = zone,
            **pipeline_args
        )

        corrected_ensembles[zone] = test["Corrected ensembles"]
        lstm_quantiles[zone] = test["Corrected ensembles original space"]
        taqr_quantiles[zone] = test["TAQR results"]
        actuals[zone] = test["Actuals"]
        beta_parameters[zone] = test["Beta"]


corrected_ensembles = pd.concat(corrected_ensembles.values(), axis = 1, keys = zones)
taqr_quantiles = pd.concat(taqr_quantiles.values(), axis = 1, keys = zones)
actuals = pd.concat(actuals.values(), axis = 1, keys = zones)
beta_parameters = pd.concat(beta_parameters.values(), axis = 1, keys = zones)
lstm_quantiles = pd.concat(lstm_quantiles.values(), axis = 1, keys = zones)


#%%
corrected_ensembles.to_csv(save_path / "corrected_ensembles.csv")
corrected_ensembles.to_pickle(save_path / "corrected_ensembles.pkl")

taqr_quantiles.to_csv(save_path / "taqr_quantiles.csv")
taqr_quantiles.to_pickle(save_path / "taqr_quantiles.pkl")

actuals.to_csv(save_path / "actuals.csv")
actuals.to_pickle(save_path / "actuals.pkl")

beta_parameters.to_csv(save_path / "beta_parameters.csv")
beta_parameters.to_pickle(save_path / "beta_parameters.pkl")

lstm_quantiles.to_csv(save_path / "lstm_quantiles.csv")
lstm_quantiles.to_pickle(save_path / "lstm_quantiles.pkl")

