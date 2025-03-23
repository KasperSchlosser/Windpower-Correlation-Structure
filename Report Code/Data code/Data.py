# This script runs NABQR on the data thens saves.
# should be run sparingly as fitting the lstm take quite a while

import pandas as pd
from pathlib import Path

def cleantz(ensemble):
    #drops timezones from the ensemble data
    #then drops the duplicated timeslot
    #not best solution
    
    ensemble.index = ensemble.index.to_series().apply(lambda x: x.tz_localize(None)).values
    ensemble = ensemble[~ensemble.index.duplicated()]
    
    return ensemble

PATH = Path.cwd()
load_path = PATH.parents[1] / "Data" / "Raw Data" 
save_path = PATH.parents[1] / "Data" 

#%% Load Data
# Finds ensembles and observations

ensembles = list((load_path / "Ensembles").glob("*pkl"))
observations = list((load_path / "Observations").glob("*pkl"))

raw_ensembles = {}
raw_observations = {}
dfs_cleaned = {}
for ens, obs in zip(ensembles, observations):
    
    print(ens,obs)
    
    zone = "-".join(ens.stem.split("_")[:2])
    
    ens = pd.read_pickle(ens)
    obs = pd.read_pickle(obs)
    obs.name = "Observed"
    
    raw_ensembles[zone] = ens
    raw_observations[zone] = obs
    
    #remove tz from ensemble and drop the missing hours from observation
    ens = cleantz(ens)
    obs = obs.dropna()
    
    dfs_cleaned[zone] = pd.concat((ens,obs), axis = 1, keys = ["Ensembles", "Observed"])
    
#join into big table
data_cleaned = pd.concat(dfs_cleaned, axis = 1, keys = dfs_cleaned.keys())
raw_ensembles = pd.concat(raw_ensembles, axis = 1, keys = raw_ensembles.keys())
raw_observations = pd.concat(raw_observations, axis = 1, keys = raw_observations.keys())


data_cleaned.to_pickle(save_path / "Data" / "Cleaned Data.pkl")
data_cleaned.to_csv(save_path / "Data" / "Cleaned Data.csv")

raw_ensembles.to_pickle(save_path / "Data" / "raw_ensembles.pkl")
raw_ensembles.to_csv(save_path / "Data" / "raw_ensembles.csv")

raw_observations.to_pickle(save_path / "Data" / "raw_observations.pkl")
raw_observations.to_csv(save_path / "Data" / "raw_observations.csv")
