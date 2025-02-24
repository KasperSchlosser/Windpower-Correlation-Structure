# This script runs NABQR on the data thens saves.
# should be run sparingly as fitting the lstm take quite a while

import numpy as np
import pandas as pd
import nabqr

from pathlib import Path

def cleantz(ensemble):
    #drops timezones from the ensemble data
    #then drops the duplicated timeslot
    #not best solution
    
    ensemble.index = ensemble.index.to_series().apply(lambda x: x.tz_localize(None)).values
    ensemble = ensemble[~ensemble.index.duplicated()]
    
    return ensemble


#%% parameters

pipeline_args = {
    "training_size": 0.8,
    "epochs": 50,
    "quantiles_taqr": np.arange(0.01,1,0.01)
    }

PATH = Path()

#%% Load Data
# Finds ensembles and observations

ensembles = list((PATH / "Data" / "Ensembles").glob("*"))
observations = list((PATH / "Data" / "Observations").glob("*wind*"))


dfs = {}
for ens, obs in zip(ensembles, observations):
    
    print(ens,obs)
    
    zone = "-".join(ens.stem.split("_")[:2])
    
    ens = pd.read_pickle(ens)
    obs = pd.read_pickle(obs)
    
    #remove tz from ensemble and drop the missing hours from observation
    ens = cleantz(ens)
    obs = obs.dropna()
    
    obs.name = "Observed"
        
    dfs[zone] = pd.concat((ens,obs), axis = 1, keys = ["Ensembles", "Observed"])
    
#join into big table
full_data = pd.concat(dfs, axis = 1, keys = dfs.keys())
full_data.to_pickle(PATH / "Data" / "Full Data.pkl")
full_data.to_csv(PATH / "Data" / "Full Data.csv")

#%% Run nabqr

for zone in full_data.columns.get_level_values(0).unique():
    
        _, estimated_quantiles, actual, _, _ = nabqr.pipeline(
            full_data[zone,"Ensembles"],
            full_data[zone,"Observed"].values.squeeze(),
            name = zone,
            **pipeline_args
        )
        
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
        
#%% Tail propbabilities


_, estimated_quantiles, actual, _, _ = nabqr.pipeline(
    full_data["DK1-offshore","Ensembles"],
    full_data["DK1-offshore","Observed"].values.squeeze(),
    quantiles_taqr = np.arange(0.99, 1,0.0001),
    epochs = 200,
    name = "Tail_estimates"
)


estimated_quantiles.columns = [f'{x:.04f}' for x in np.arange(0.99, 1,0.0001)]
actual.name = "Observed"

tmp = pd.concat((estimated_quantiles, actual), axis = 1, keys = ["Quantiles", "Observed"])
tmp.to_csv(PATH / "Data" / "NABQR Results" / ( "tails.csv" ))
tmp.to_pickle(PATH / "Data" / "NABQR Results" / ("tails.pkl" ))
        
        
        
    

