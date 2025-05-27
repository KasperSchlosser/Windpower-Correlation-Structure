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

global_params = {"n_sim": 250, "horizon": 24, "alpha": 0.1, "sided": 2}


# maybe ARMA need to just be made separately
# "ARMA": {
#     "basis": "actuals",
#     "quantile model": None,
#     "quantile params": {},
#     "model": nabqra.correlation.sarma,
#     "params": {"order": (4, 0, 0), "trend": "c"},
# },
models = {
    "Ensemble": {"basis": "Ensemble", "model": nabqra.correlation.dummy, "params": {}},
    "Feature": {"basis": "Feature", "model": nabqra.correlation.dummy, "params": {}},
    "Feature +  SARMA": {
        "basis": "Feature",
        "model": nabqra.correlation.sarma,
        "params": {"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 24)},
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
resid_onestep = pd.DataFrame(
    index=pd.MultiIndex.from_product((zones, test_index), names=["Zone", "Time"]), columns=["resid"]
)


def params_to_pd(summary):
    data = np.array(summary.tables[1].data)
    return pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])


model_params = {}

for zone, model in scores.index.unique():
    print(zone, model)

    params = models[model]
    actuals = obs.loc[zone]
    est_quantiles = nabqra.misc.fix_quantiles(basis_quantiles.loc[zone, params["basis"]], *zone_lims[zone])

    quantile_model = nabqra.quantiles.spline_model(quantiles, *zone_lims[zone])
    correlation_model = params["model"](**params["params"], **global_params)
    correlation_model_onestep = params["model"](**params["params"], horizon=1, n_sim=3)

    pipeline = nabqra.pipeline.pipeline(correlation_model, quantile_model)

    res = pipeline.run(est_quantiles, actuals, train_index)

    forecast_res.loc[zone, model, :] = res[0].values[len(train_index) :, :]

    est = res[0].loc[test_index, idx["Original", "Estimate"]]
    sim = res[1].loc[test_index, "Original"]

    scores.loc[idx[zone, model], "MAE":"VarS"] = nabqra.scoring.calc_scores(
        actuals.loc[test_index].values, est.values, sim.values
    )

    if hasattr(pipeline.correlation_model, "get_params"):
        # get params and onestep
        model_params[zone] = params_to_pd(pipeline.correlation_model.get_params())

        tmp = correlation_model.modelres.apply(res[0]["Normal"]["Observation"], refit=False)

        resid_onestep.loc[idx[zone, :]] = tmp.resid[test_index].values


model_params = pd.concat(model_params, names=["Zone"])
# model_params = model_params[["coef", "std err"]]
# model_params.columns = ["Estimated Value", "Standard Error"]
model_params = model_params.stack().unstack(level=1)


# %% save data
forecast_res.to_pickle(save_path / "Forecast.pkl")
forecast_res.to_csv(save_path / "Forecast.csv")

resid_onestep.to_pickle(save_path / "Residuals onestep.pkl")
resid_onestep.to_csv(save_path / "Residuals onestep.csv")

# scores.to_pickle(save_path / "Forecast scores.pkl")
# scores.to_csv(save_path / "Forecast scores.csv")

model_params.to_pickle(save_path / "Model Params.pkl")
model_params.to_csv(save_path / "Model Params.csv")
