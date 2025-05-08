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
from statsmodels.othermod.betareg import BetaModel
from sklearn.linear_model import QuantileRegressor

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Basis"
save_path = PATH / "Data" / "Basis"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_lims = pd.DataFrame(parameters["Zone-Limits"], index=["Min", "Max"]).T
    quantiles = parameters["Quantiles"]
    quantiles_str = [f"{q:.2f}" for q in quantiles]

obs = pd.read_pickle(load_path / ".." / "Data" / "cleaned_observations.pkl")


index = obs.index.unique(1)
train_index = index[: parameters["train_size"]]
test_index = index[parameters["train_size"] :]

features = pd.read_pickle(load_path / "Features.pkl")
nabqr_features = pd.read_pickle(load_path / "Basis quantiles.pkl").xs("Original - Identity", level=1)

results = pd.read_pickle(load_path / "Basis Quantiles.pkl")

plt.close("all")
# %% comparison between quantiles and reduced basis
# maybe not?

features.loc[idx[:, :], "const"] = 1
nabqr_features.loc[idx[:, :], "const"] = 1

reg_res = pd.DataFrame(
    columns=["0.01", "0.50", "0.99"],
    index=pd.MultiIndex.from_product((zones, ("Quantile Basis", "Simple Basis"), index)),
    dtype=np.float64,
)


for zone in zones:
    print(zone)
    for q in ["0.01", "0.50", "0.99"]:
        print(q)
        model_small = sm.QuantReg(obs.loc[zone, train_index], features.loc[(zone, train_index), :]).fit(float(q))
        print("small done")
        model_big = sm.QuantReg(obs.loc[zone, train_index], nabqr_features.loc[(zone, train_index), :]).fit(float(q))
        print("big done")

        reg_res.loc[idx[zone, "Simple Basis", :], q] = model_small.predict(features.loc[zone]).values
        reg_res.loc[idx[zone, "Quantile Basis", :], q] = model_big.predict(nabqr_features.loc[zone]).values
# %%
for zone in zones:
    fig, ax = plt.subplots()
    ax.plot(test_index, reg_res.loc[idx[zone, "Simple Basis", test_index], "0.01"])
    ax.plot(test_index, reg_res.loc[idx[zone, "Quantile Basis", test_index], "0.01"])


# %% add beta and taqr quantiles


# for zone in zones:

#     feat = features.loc[zone]
#     scale = zone_lims.loc[zone, "Max"]
#     obs_scaled = obs.loc[zone] / scale

#     data = pd.concat([obs_scaled, feat], axis=1)

#     model = BetaModel.from_formula(
#         formula="Observed ~ Feature + I(Feature**2) + I(Feature**3)",
#         exog_precision_formula="Feature + I(Feature**2) + I(Feature**3)",
#         data=data.loc[train_index],
#     ).fit()

#     train_dist = model.get_distribution(feat.loc[train_index])
