import pathlib
import tomllib

import pandas as pd
import numpy as np
import statsmodels.api as sm
import nabqra

from pandas import IndexSlice as idx


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
ensembles = pd.read_pickle(load_path / ".." / "Data" / "cleaned_ensembles.pkl")

index = obs.index.unique(1)
train_index = index[: parameters["train_size"]]
test_index = index[parameters["train_size"] :]

features = pd.read_pickle(load_path / "Features.pkl")
nabqr_features = pd.read_pickle(load_path / "Basis normalised.pkl").xs("NABQR", level=1)

features.loc[idx[:, :], "const"] = 1
nabqr_features.loc[idx[:, :], "const"] = 1

quantile_res = pd.read_pickle(load_path / "NN quantiles.pkl")

rng = np.random.default_rng(42)
n_sim = 250

# %% comparison between using quantiles and reduced basis

reg_res = pd.DataFrame(
    columns=["0.01", "0.50", "0.60"],
    index=pd.MultiIndex.from_product(
        (zones, ("Quantile Basis", "Small Basis"), index), names=["Zone", "Basis", "Time"]
    ),
    dtype=np.float64,
)


for zone in zones:
    print(zone)
    for q in ["0.01", "0.50", "0.60"]:
        print(q)
        model_small = sm.QuantReg(obs.loc[zone, train_index], features.loc[(zone, train_index), :]).fit(float(q))
        model_big = sm.QuantReg(obs.loc[zone, train_index], nabqr_features.loc[(zone, train_index), :]).fit(float(q))

        reg_res.loc[idx[zone, "Small Basis", :], q] = model_small.predict(features.loc[zone]).values
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


# %% add ensemble estimates

ensembles_quant = ensembles.quantile(quantiles, axis=1).T
ensembles_quant.columns = quantiles_str
ensembles_quant = pd.concat([ensembles_quant], keys=["Ensemble"], names=["Model"]).swaplevel(0, 1)
quantile_res = pd.concat((quantile_res, ensembles_quant)).sort_index()

# %% TAQR original
# initialise taqr with 500 data points, max it can take without problems
# nope it takes so long to run
# just run on the last 10000 data points this includes the entire test set

taqr_quant = pd.DataFrame(
    index=pd.MultiIndex.from_product((zones, index), names=["Zone", "Time"]),
    columns=quantile_res.columns,
    dtype=np.float64,
)

init = 500
for zone in zones:

    tmp = nabqra.taqr.run_taqr(
        nabqr_features.loc[zone].iloc[-10000:],
        obs.loc[zone].iloc[-10000:],
        quantiles,
        init,
        len(obs.loc[zone].iloc[-10000:]),
        init,
    )
    taqr_quant.loc[idx[zone, index[-10000 + 502 :]], :] = np.stack(tmp[0], axis=1)

taqr_quant = pd.concat([taqr_quant], keys=["NABQR - TAQR"], names=["Model"]).swaplevel(0, 1)
quantile_res = pd.concat((quantile_res, taqr_quant)).sort_index()

# %% find pseudo Residuals

residuals = pd.DataFrame(index=quantile_res.index, columns=["CDF", "Normal"])
residuals = residuals.join(obs.rename("Original"))

for (zone, model), est_quant in quantile_res.groupby(level=(0, 1)):
    print(zone, model)
    # fix quantiles if bad
    est_quant = nabqra.misc.fix_quantiles(est_quant, *zone_lims[zone])

    dist = nabqra.quantiles.spline_model(quantiles, *zone_lims[zone])

    tmp = dist.transform(est_quant.values, residuals.loc[idx[zone, model, :], "Original"].values)
    residuals.loc[idx[zone, model, :], "Normal"] = tmp[0]
    residuals.loc[idx[zone, model, :], "CDF"] = tmp[1]


# %% calculate scores
# should also calculate quantile score
scores = pd.DataFrame(
    index=quantile_res.droplevel(2).index.unique(), columns=["MAE", "RMSE", "CRPS", "VarS"], dtype=np.float64
)

for (zone, model), est_quant in quantile_res.groupby(level=(0, 1)):
    print(zone, model)
    # fix quantiles if bad
    est_quant = nabqra.misc.fix_quantiles(est_quant.loc[:, :, test_index], *zone_lims[zone])
    dist = nabqra.quantiles.spline_model(quantiles, *zone_lims[zone])

    actuals = obs.loc[zone, test_index]
    prediction = est_quant["0.50"]
    sim = dist.make_sim(est_quant.values, n_sim=n_sim, random_state=rng)

    print("sim done")

    scores.loc[idx[zone, model], :] = nabqra.scoring.calc_scores(actuals.values, prediction, sim)

# %% save

quantile_res.to_csv(save_path / "Basis quantiles.csv")
quantile_res.to_pickle(save_path / "Basis quantiles.pkl")

scores.to_csv(save_path / "Basis scores.csv")
scores.to_pickle(save_path / "Basis scores.pkl")

residuals.to_csv(save_path / "Basis residuals.csv")
residuals.to_pickle(save_path / "Basis residuals.pkl")


# %% Alignment Comparison

data = pd.read_pickle(save_path / "Basis quantiles.pkl").loc[idx[:, ["Ensemble", "NABQR - TAQR"], test_index]]
scores = pd.DataFrame(index=data.droplevel(2).index.unique(), columns=["MAE", "RMSE", "CRPS", "VarS"], dtype=np.float64)
scores_mis = pd.DataFrame(
    index=data.droplevel(2).index.unique(), columns=["MAE", "RMSE", "CRPS", "VarS"], dtype=np.float64
)

quantile_loss = nabqra.scoring.Quantileloss(quantiles)
for (zone, model), est_quant in data.groupby(level=(0, 1)):
    print(zone, model)
    # fix quantiles if bad
    est_quant = nabqra.misc.fix_quantiles(est_quant.loc[:, :, test_index], *zone_lims[zone])
    dist = nabqra.quantiles.spline_model(quantiles, *zone_lims[zone])

    actuals = obs.loc[zone, test_index]
    prediction = est_quant["0.50"]
    sim = dist.make_sim(est_quant.values, n_sim=n_sim, random_state=rng)

    print("sim done")

    scores.loc[idx[zone, model], :] = nabqra.scoring.calc_scores(actuals.values, prediction, sim)

    ix = np.arange(len(test_index))[1:-1]
    # add misalignment
    actuals = actuals[2:]
    match model:
        case "NABQR - TAQR":
            ix = np.arange(len(test_index))[2:]
        case "Ensemble":
            ix = np.arange(len(test_index))[1:-1]

    scores_mis.loc[idx[zone, model]] = nabqra.scoring.calc_scores(actuals.values, prediction.iloc[ix], sim[ix, :])

scores = pd.concat([scores, scores_mis], keys=["Correct alignment", "Misaligned"], names=["Alignment"])

scores.to_csv(save_path / "Alignment scores.csv")
scores.to_pickle(save_path / "Alignment scores.pkl")
