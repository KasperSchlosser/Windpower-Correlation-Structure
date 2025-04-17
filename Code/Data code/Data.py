# This script runs NABQR on the data thens saves.
# should be run sparingly as fitting the lstm take quite a while

import pandas as pd
import tomllib
from pathlib import Path

def cleantz(ensemble):
    # drops timezones from the ensemble data
    # then drops the duplicated timeslot
    # might not be the best solution
    
    ensemble.index = ensemble.index.to_series().apply(lambda x: x.tz_localize(None)).values
    ensemble = ensemble[~ensemble.index.duplicated()]
    ensemble.dropna()
    
    return ensemble

PATH = Path.cwd().parents[1]
load_path = PATH / "Data" / "Raw Data" 
save_path = PATH / "Data" / "Data"

# The observed production sometimes goes negative, this is an error
# values of 0 would also give problems
# this value is for practical purpose 0
low_value = 0.01

with open(PATH / "Settings" / "zone limits.toml", "rb") as f:
    zone_lims = tomllib.load(f)

#%% Load Data
# Finds ensembles and observations

ensembles = list((load_path / "Ensembles").glob("*pkl"))
observations = list((load_path / "Observations").glob("*pkl"))

raw_ensembles = {}
raw_observations = {}
cleaned_ensembles =  {}
cleaned_observations = {}
normalised_ensembles = {}
normalised_observations = {}

for ens, obs in zip(ensembles, observations):
    
    zone = "-".join(ens.stem.split("_")[:2])
    print(zone)
    
    ens = pd.read_pickle(ens)
    obs = pd.read_pickle(obs)
    obs.name = "Observed"
    
    # need original data for plots
    raw_ensembles[zone] = ens.copy(deep = True)
    raw_observations[zone] = obs.copy(deep = True)
    
    #remove tz from ensemble and drop the missing hours from observation
    ens = cleantz(ens)
    
    # change problematic observations
    obs[obs < low_value] = low_value

    cleaned_ensembles[zone] = ens
    cleaned_observations[zone] = obs
    
    normalised_ensembles[zone] = (ens - ens.min()) / (ens.max() - ens.min())
    normalised_observations[zone] = (obs - zone_lims[zone][0]) / (zone_lims[zone][1] - zone_lims[zone][0])

cleaned_ensembles = pd.concat(cleaned_ensembles, axis = 1, keys = cleaned_ensembles.keys())
cleaned_observations = pd.concat(cleaned_observations, axis = 1, keys = cleaned_observations.keys())
cleaned_observations.dropna(inplace = True)

raw_ensembles = pd.concat(raw_ensembles, axis = 1, keys = raw_ensembles.keys())
raw_observations = pd.concat(raw_observations, axis = 1, keys = raw_observations.keys())

normalised_ensembles = pd.concat(normalised_ensembles, axis = 1, keys = normalised_ensembles.keys())
normalised_observations = pd.concat(normalised_observations, axis = 1, keys = normalised_observations.keys())

cleaned_ensembles.to_pickle(save_path / "cleaned_ensembles.pkl")
cleaned_ensembles.to_csv(save_path / "cleaned_ensembles.csv")

cleaned_observations.to_pickle(save_path / "cleaned_observations.pkl")
cleaned_observations.to_csv(save_path / "cleaned_observations.csv")

raw_ensembles.to_pickle(save_path / "raw_ensembles.pkl")
raw_ensembles.to_csv(save_path / "raw_ensembles.csv")

raw_observations.to_pickle(save_path / "raw_observations.pkl")
raw_observations.to_csv(save_path / "raw_observations.csv")

normalised_ensembles.to_pickle(save_path / "normalised_ensembles.pkl")
normalised_ensembles.to_csv(save_path / "normalised_ensembles.csv")

normalised_observations.to_pickle(save_path / "normalised_observations.pkl")
normalised_observations.to_csv(save_path / "normalised_observations.csv")
