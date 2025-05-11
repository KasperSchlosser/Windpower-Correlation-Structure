import pathlib
import tomllib

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm


from pandas import IndexSlice as idx

from nabqra.scoring import Quantileloss
from nabqra.misc import fix_quantiles
from nabqra.quantiles import spline_model
from statsmodels.othermod.betareg import BetaModel

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Basis"
save_path = PATH / "Data" / "Basis"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_lims = parameters["Zone-Limits"]
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

quantile_res = pd.read_pickle(load_path / "Basis Quantiles.pkl")

# %% comparison between using quantiles and reduced basis


reg_res = pd.DataFrame(
    columns=["0.01", "0.50", "0.60"],
    index=pd.MultiIndex.from_product(
        (zones, ("Quantile Basis", "Simple Basis"), index), names=["Zone", "Basis", "Time"]
    ),
    dtype=np.float64,
)

res_scores = pd.DataFrame()

for zone in zones:
    print(zone)
    for q in ["0.01", "0.50", "0.60"]:
        print(q)
        model_small = sm.QuantReg(obs.loc[zone, train_index], features.loc[(zone, train_index), :]).fit(float(q))
        model_big = sm.QuantReg(obs.loc[zone, train_index], nabqr_features.loc[(zone, train_index), :]).fit(float(q))

        reg_res.loc[idx[zone, "Simple Basis", :], q] = model_small.predict(features.loc[zone]).values
        reg_res.loc[idx[zone, "Quantile Basis", :], q] = model_big.predict(nabqr_features.loc[zone]).values

reg_res = reg_res.join(obs, on=["Zone", "Time"])
res_test = reg_res.loc[idx[:, :, test_index], :]

res_scores = (res_test["0.50"] - res_test["Observed"]).abs().groupby(level=(0, 1)).mean().to_frame(name="MAE")
res_scores["Crossings"] = (res_test["0.50"] > res_test["0.60"]).groupby(level=(0, 1)).sum()
res_scores["Violations"] = (res_test["0.01"] < 0.0).groupby(level=(0, 1)).sum()
res_scores = res_scores.unstack(level="Basis")
res_scores.loc["Total"] = res_scores.sum()


res_scores.to_csv(save_path / "Basis comparison scores.csv")
res_scores.to_pickle(save_path / "Basis comparison scores.pkl")

reg_res.to_csv(save_path / "Basis comparison.csv")
reg_res.to_pickle(save_path / "Basis comparison.pkl")


# %% find pseudo Residuals

residuals = pd.DataFrame(index=quantile_res.index, columns=["CDF", "Normal"])
residuals = residuals.join(obs.rename("Original"))


for (zone, model), est_quant in quantile_res.groupby(level=(0, 1)):
    print(zone, model)
    # fix quantiles if bad
    # residuals.loc[idx[zone, model, :], "Original"] = obs.loc[zone].values
    est_quant = fix_quantiles(est_quant, *zone_lims[zone])

    dist = spline_model(quantiles, *zone_lims[zone])

    tmp = dist.transform(est_quant.values, residuals.loc[idx[zone, model, :], "Original"].values)
    residuals.loc[idx[zone, model, :], "Normal"] = tmp[0]
    residuals.loc[idx[zone, model, :], "CDF"] = tmp[1]


# %% add taqr and beta estimates
# maybe not taqr, is quite problematic

beta_quant = pd.DataFrame(
    index=pd.MultiIndex.from_product((zones, index), names=["Zone", "Time"]),
    columns=quantile_res.columns,
    dtype=np.float64,
)
beta_residual = pd.DataFrame(index=beta_quant.index, columns=["Original", "CDF", "Normal"], dtype=np.float64)

for zone in zones:

    feat = features.loc[zone]
    scale = zone_lims[zone][1]
    obs_scaled = obs.loc[zone] / scale

    model = BetaModel(obs_scaled.loc[train_index], feat.loc[train_index], feat.loc[train_index]).fit()

    dist = model.get_distribution(feat, feat)

    beta_quant.loc[zone, :] = dist.ppf(quantiles[:, np.newaxis]).T * scale

    beta_residual.loc[idx[zone, :], "Original"] = obs.loc[zone].values
    beta_residual.loc[idx[zone, :], "CDF"] = dist.cdf(obs_scaled)
    beta_residual.loc[idx[zone, :], "Normal"] = stats.norm().ppf(beta_residual.loc[idx[zone, :], "CDF"])


# add model name
beta_quant = pd.concat([beta_quant], keys=["Beta Regression"], names=["Model"]).swaplevel(0, 1)
beta_residual = pd.concat([beta_residual], keys=["Beta Regression"], names=["Model"]).swaplevel(0, 1)

residuals = pd.concat((residuals, beta_residual)).sort_index()
quantile_res = pd.concat((quantile_res, beta_quant)).sort_index()

# %%
