import pathlib
import tomllib

import pandas as pd
import numpy as np

import nabqra

from pandas import IndexSlice as idx


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Basis"
save_path = PATH / "Data" / "Autocorrelation"


with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_lims = parameters["Zone-Limits"]
    quantiles = np.array(parameters["Quantiles"])
    quantiles_str = [f"{q:.2f}" for q in quantiles]

basis_quantiles = pd.read_pickle(load_path / "Basis quantiles.pkl")
obs = pd.read_pickle(load_path / ".." / "Data" / "cleaned_observations.pkl")

index = obs.index.unique(1)
train_index = index[: parameters["train_size"]]
test_index = index[parameters["train_size"] :]


# %% models

global_params = {"n_sim": 10, "burnin": 240, "horizon": 24, "alpha": 0.1, "sided": 2}


# maybe ARMA need to just be made separately
# "ARMA": {
#     "basis": "actuals",
#     "quantile model": None,
#     "quantile params": {},
#     "model": nabqra.correlation.sarma,
#     "params": {"order": (4, 0, 0), "trend": "c"},
# },
models = {
    "Ensemble": {
        "basis": "Ensemble",
        "model": nabqra.correlation.dummy,
        "params": {},
    },
    "Ensemble + SARMA": {
        "basis": "Ensemble",
        "model": nabqra.correlation.sarma,
        "params": {"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 24)},
    },
    "Full Model +  SARMA": {
        "basis": "Feature",
        "model": nabqra.correlation.sarma,
        "params": {"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 24)},
    },
    "Full Model": {
        "basis": "Simple",
        "model": nabqra.correlation.dummy,
        "params": {},
    },
}


# %% make data frames

forecast_res = pd.DataFrame(
    index=pd.MultiIndex.from_product((zones, models.keys(), test_index), names=["Zone", "Model", "Time"]),
    columns=pd.MultiIndex.from_product((["Original", "CDF", "Normal"], ["Observation", "Estimate", "Lower", "Upper"])),
)
scores = pd.DataFrame(
    index=forecast_res.droplevel(2).index.unique(), columns=["MAE", "RMSE", "CRPS", "VarS"], dtype=np.float64
)


for zone, model in scores.index.unique():
    print(zone, model)

    params = models[model]
    actuals = obs.loc[zone]
    est_quantiles = nabqra.misc.fix_quantiles(basis_quantiles.loc[zone, params["basis"]], *zone_lims[zone])

    quantile_model = nabqra.quantiles.spline_model(quantiles, *zone_lims[zone])
    correlation_model = params["model"](**params["params"], **global_params)

    pipeline = nabqra.pipeline.pipeline(correlation_model, quantile_model)

    res = pipeline.run(est_quantiles, actuals, train_index)

    forecast_res.loc[zone, model, :] = res[0].values[len(train_index) :, :]

    est = res[0].loc[test_index, idx["Original", "Estimate"]]
    sim = res[1].loc[test_index, "Original"]

    scores.loc[idx[zone, model], "MAE":"VarS"] = nabqra.scoring.calc_scores(
        actuals.loc[test_index].values, est.values, sim.values
    )

# %% save data

# forecast.to_pickle(save_path / "forecast results.pkl")
# forecast.to_csv(save_path / "forecast results.csv")

# simulation.to_pickle(save_path / "simulation results.pkl")
# # simulation.to_csv(save_path / "simulation results.csv") #maybe not save this as cvs, currently 1.6 gb, woops
