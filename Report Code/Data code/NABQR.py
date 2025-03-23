# This script runs NABQR on the data thens saves.
# should be run sparingly as fitting the lstm take quite a while

import numpy as np
import pandas as pd
import nabqr

from pathlib import Path


quantiles = np.arange(0.01, 1,0.25)
quantiles_str = [f'{x:.02f}' for x in quantiles]
pipeline_args = {
    "training_size": 0.8,
    "epochs": 2,
    "quantiles_taqr": quantiles 
    }

PATH = Path.cwd()
load_path = PATH.parents[1] / "Data" / "Data"
save_path = PATH.parents[1] / "Data" / "NABQR"

data = pd.read_pickle(load_path / "Cleaned Data.pkl")

#%% nabqr


corrected_ensembles = {}
estimated_quantiles = {}
actuals = {}
beta_parameters = {}
original_ensembles = {}
res = {}
for zone in data.columns.get_level_values(0).unique():
    
        c_ens, est_q, act, beta, orig  = nabqr.pipeline(
            data[zone,"Ensembles"],
            data[zone,"Observed"].values.squeeze(),
            name = zone,
            **pipeline_args
        )
        
        c_ens.columns = ["Corrected Ensemble " + x for x in c_ens.columns]
        est_q.coloumns = quantiles_str
        act.name = "Value"
        
        beta.con
        
        
        res[zone] = pd.concat(tmp, axis = 1, keys = ["Original Ensemble", "Corrected Ensembles", "Estimated Quantiles", "Observed Value", "Beta Parameter"])
        


#%%
"""
        # Needs to be change if finer quantiles are used
        estimated_quantiles.columns = [f'{x:.02f}' for x in pipeline_args["quantiles_taqr"]]
        actual.name = "Observed"
        
        tmp = pd.concat((estimated_quantiles, actual), axis = 1, keys = ["Quantiles", "Observed"])
        tmp.to_csv(PATH / "Data" / "NABQR Results" / (zone + ".csv" ))
        tmp.to_pickle(PATH / "Data" / "NABQR Results" / (zone + ".pkl" ))

#%% Make df with all data

datafiles = list((PATH / "Data" / "NABQR Results").glob("DK*pkl"))
files = [pd.read_pickle(f) for f in datafiles]
names = [f.stem for f in datafiles]

data_big = pd.concat(files, axis = 1, keys = names)

data_big.to_pickle(PATH / "Data" / ("NABQR_results_full.pkl" ))
data_big.to_csv(PATH / "Data" / ("NABQR_results_full.csv" ))
        """