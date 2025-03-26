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
    "epochs": 200,
    "quantiles_taqr": np.arange(1,100) / 100 
    }

PATH = Path()

#%% Load Data
# Finds ensembles and observations

ensembles = list((PATH.parent / "Data" / "Raw Data" / "Ensembles").glob("*pkl"))
observations = list((PATH.parent / "Data" / "Raw Data" / "Observations").glob("*pkl"))


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


        
    

