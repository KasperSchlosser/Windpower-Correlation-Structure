import pathlib
import tomllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pandas import IndexSlice as idx
import nabqra.taqr as taqr
from nabqra.scoring import Quantileloss
from statsmodels.othermod.betareg import BetaModel

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Basis"
save_path = PATH / "Data" / "Basis"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_lims = pd.DataFrame(parameters["Zone-Limits"], index=["Min", "Max"]).T
    quantiles = np.array(parameters["Quantiles"])
    quantiles_str = [f"{q:.2f}" for q in quantiles]

obs = pd.read_pickle(load_path / ".." / "Data" / "cleaned_observations.pkl")


index = obs.index.unique(1)
train_index = index[: parameters["train_size"]]
test_index = index[parameters["train_size"] :]

features = pd.read_pickle(load_path / "Features.pkl")
nabqr_features = pd.read_pickle(load_path / "Basis normalised.pkl").xs("Original - Identity", level=1)

features.loc[idx[:, :], "const"] = 1
nabqr_features.loc[idx[:, :], "const"] = 1

results = pd.read_pickle(load_path / "Basis Quantiles.pkl")

# %% comparison between quantiles and reduced basis
# maybe not?

reg_res = pd.DataFrame(
    columns=["0.01", "0.50", "0.90"],
    index=pd.MultiIndex.from_product((zones, ("Quantile Basis", "Simple Basis"), index)),
    dtype=np.float64,
)


for zone in zones:
    print(zone)
    for q in ["0.01", "0.50", "0.90"]:
        print(q)
        model_small = sm.QuantReg(obs.loc[zone, train_index], features.loc[(zone, train_index), :]).fit(float(q))
        print("small done")
        model_big = sm.QuantReg(obs.loc[zone, train_index], nabqr_features.loc[(zone, train_index), :]).fit(float(q))
        print("big done")

        reg_res.loc[idx[zone, "Simple Basis", :], q] = model_small.predict(features.loc[zone]).values
        reg_res.loc[idx[zone, "Quantile Basis", :], q] = model_big.predict(nabqr_features.loc[zone]).values


# %% add taqr and beta estimates

beta = pd.DataFrame(index=results.index.droplevel(1).unique(), columns=results.columns, dtype=np.float64)

for zone in zones:

    feat = features.loc[zone]
    scale = zone_lims.loc[zone, "Max"]
    obs_scaled = obs.loc[zone] / scale

    model = BetaModel(obs_scaled.loc[train_index], feat.loc[train_index], feat.loc[train_index]).fit()

    dist = model.get_distribution(feat, feat)
    beta.loc[zone, :] = dist.ppf(quantiles[:, np.newaxis]).T * scale

# add model name
beta = pd.concat([beta], keys=["Beta Regression"]).swaplevel(0, 1)
