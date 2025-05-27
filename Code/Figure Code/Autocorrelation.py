import pathlib
import tomllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nabqra.plotting as nplt
from pandas import IndexSlice as idx

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Autocorrelation"
save_path = PATH / "Results" / "Autocorrelation"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]

scores = pd.read_pickle(load_path / "Forecast scores.pkl")
preds = pd.read_pickle(load_path / "Forecast.pkl").astype(np.float64)
resids_onestep = pd.read_pickle(load_path / "Residuals onestep.pkl").astype(np.float64)
observations = pd.read_pickle(load_path / ".." / "Data" / "cleaned_observations.pkl")
params = pd.read_pickle(load_path / "Model Params.pkl")

models = preds.index.unique(1)


# %% Example
zone = "DK1-onshore"
obs = observations.loc[zone]
period = [np.datetime64("2023-10-17"), np.datetime64("2023-11-02")]

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
axes = axes.ravel()

for model, ax in zip(models, axes):

    df = preds.loc[idx[zone, model], "Original"]
    nplt.band_plot(df.index, df["Estimate"], df["Lower"], df["Upper"], ax=ax, alpha=0.3)

    ax.scatter(obs.index, obs, color="black", s=1)
    ax.set_xlim(period)
    ax.set_ylim([-10, 3800])
    ax.set_title(model)

fig.savefig(save_path / "Figures" / "model_example")

# %% Sarma in 3 spaces

zone = "DK1-onshore"
obs = observations.loc[zone]
period = [np.datetime64("2023-10-17"), np.datetime64("2023-11-02")]

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
axes = axes.ravel()

for space, ax in zip(["Original", "CDF", "Normal"], axes):

    df = preds.loc[idx[zone, model]]
    nplt.band_plot(df.index, df[space, "Estimate"], df[space, "Lower"], df[space, "Upper"], ax=ax, alpha=0.3)

    ax.scatter(df.index, df[space, "Observation"], color="black", s=1)
    ax.set_xlim(period)
    ax.set_title(space)

fig.savefig(save_path / "Figures" / "space_example")

# %% Residual

for zone, df in resids_onestep.groupby("Zone"):
    print(zone)
    figs, _ = nplt.diagnostic_plots(
        df.values.squeeze() / df.std().values,
        df.index.get_level_values(1),
        save_path=save_path / "Figures" / "Residuals" / f"{zone}",
    )
    figs[0].suptitle(f"{zone}")


# %% param tabel
tmp = params.loc[idx[:, ["coef", "std err"]], :]
tmp = params.groupby("Zone").apply(lambda x: x.T.astype(str).apply(lambda x: "$" + "\pm".join(x) + "$", 1)).T
tmp = tmp.set_index(pd.Index(["$AR_1$", "$MA_1$", "$SAR_{1,24}$", "$SAM_{1,24}$", "$\sigma^2$"]))
tmp.style.to_latex(
    save_path / "Tables" / "Params.tex",
    hrules=True,
    clines="skip-last;data",
    convert_css=True,
    position="h",
    position_float="centering",
    multicol_align="r",
    multirow_align="r",
    caption=(
        "Parameters for the sarma model in normal space in all zones there is a strong autocorrelation at lag 1. "
        "There is also a strong seasonal component in all zone except DK2-offshore",
        "Paraeters for the SARMA model ",
    ),
)

# %% Result table

avg_score = pd.concat([np.exp(np.log(scores).groupby(level=1).mean())], keys=["Geometric mean score"])
scores = pd.concat([scores, avg_score])

min_score = scores.groupby(level=0).transform("min")
gmap = np.log(scores / min_score)
min_format = np.where(scores == min_score, "font-weight: bold; font-style: italic", "")
min_format = pd.DataFrame(min_format, index=scores.index, columns=scores.columns)

(
    scores.style.format(precision=2)
    .background_gradient(cmap="Reds", vmin=0, vmax=np.log(2), axis=None, gmap=gmap)
    .apply(
        lambda x: min_format,
        axis=None,
    )
    .to_latex(
        save_path / "Tables" / "Scores.tex",
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        position="h",
        position_float="centering",
        multicol_align="r",
        multirow_align="r",
        caption=(
            "Scores for the original ensembles, NN-model, and NN-model with SARMA. "
            "The SARMA corrected model performs much better than the other models on all scores. ",
            "Scores for the autocorrelation models.",
        ),
    )
)
