import pathlib
import tomllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nabqra.plotting as nplt
from pandas import IndexSlice as idx

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Basis"
save_path = PATH / "Results" / "Basis"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]

scores = pd.read_pickle(load_path / "Basis scores.pkl")
history = pd.read_pickle(load_path / "History.pkl")
preds = pd.read_pickle(load_path / "Basis Quantiles.pkl")
resids = pd.read_pickle(load_path / "Basis Residuals.pkl").astype(np.float64)
observations = pd.read_pickle(load_path / ".." / "Data" / "cleaned_observations.pkl")

models = preds.index.unique(1)

# %% NN training

fig, axes = plt.subplots(2, 2, sharex=True, sharey=False)
axes = axes.ravel()
for ax, zone in zip(axes, zones):
    tmp = history.loc[zone, "loss"].unstack("Model")
    ax.semilogy(tmp)
    ax.legend(tmp.columns)
    ax.set_title(zone)
fig.suptitle("Train Loss")
fig.supxlabel("Epoch")
fig.supylabel("Loss")

fig.savefig(save_path / "Figures" / "Train loss")


fig, axes = plt.subplots(2, 2, sharex=True, sharey=False)
axes = axes.ravel()
for ax, zone in zip(axes, zones):
    tmp = history.loc[zone, "val_loss"].unstack("Model")
    ax.semilogy(tmp)
    ax.legend(tmp.columns)
    ax.set_title(zone)
fig.suptitle("Validation Loss")
fig.supxlabel("Epoch")
fig.supylabel("Loss")
fig.savefig(save_path / "Figures" / "Validation loss")

for model in history.index.unique("Model"):

    fig, ax = plt.subplots()
    for zone in zones:
        l = ax.semilogy(history.loc[zone, model]["loss"], label=f"{zone}")
        ax.semilogy(
            history.loc[zone, model]["val_loss"],
            color=l[0].get_color(),
            linestyle="--",
        )
    ax.semilogy(0, 1, color="black", label="Train loss")
    ax.semilogy(0, 1, color="black", linestyle="--", label="Validation loss")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.set_title(model)
    fig.savefig(save_path / "Figures" / f"{model} History")

# %% Model examples

zone = "DK1-onshore"
obs = observations.loc[zone]
period = [np.datetime64("2023-10-17"), np.datetime64("2023-11-02")]

for model in models:

    df = preds.loc[zone, model]

    fig, ax = plt.subplots()

    nplt.band_plot(df.index, df["0.50"], df["0.05"], df["0.95"], ax=ax)
    ax.scatter(obs.index, obs, color="black", s=1)
    ax.set_xlim(period)
    ax.set_ylim([-10, 3800])
    ax.set_title(model)

    fig.savefig(save_path / "Figures" / f"{model}_example")


# %% NABQR vs feature

zone = "DK1-onshore"
obs = observations.loc[zone]
period = [np.datetime64("2024-06-23"), np.datetime64("2024-07-08")]


df_nabqr = preds.loc[zone, "NABQR - TAQR"]
df_feature = preds.loc[zone, "Feature"]

fig, ax = plt.subplots()

nplt.band_plot(df_nabqr.index, df_nabqr["0.50"], df_nabqr["0.05"], df_nabqr["0.95"], ax=ax, label="NABQR")
nplt.band_plot(
    df_feature.index, df_feature["0.50"], df_feature["0.05"], df_feature["0.95"], ax=ax, label="Feature Model"
)
ax.scatter(obs.index, obs, color="black", s=1)
ax.set_xlim(period)
ax.set_title(model)
ax.legend()

fig.savefig(save_path / "Figures" / "Model Comparison")

# %% DK2 offshore

zone = "DK2-offshore"
obs = observations.loc[zone]
period = [np.datetime64("2023-03-08"), np.datetime64("2023-03-16")]


df_simple = preds.loc[zone, "Simple"]
df_feature = preds.loc[zone, "Feature"]

fig, ax = plt.subplots()

nplt.band_plot(df_simple.index, df_simple["0.50"], df_simple["0.05"], df_simple["0.95"], ax=ax, label="NABQR")
nplt.band_plot(
    df_feature.index, df_feature["0.50"], df_feature["0.05"], df_feature["0.95"], ax=ax, label="Feature Model"
)
ax.scatter(obs.index, obs, color="black", s=1)
ax.set_xlim(period)
ax.set_title(model)
ax.legend()

fig.savefig(save_path / "Figures" / "DK2-offshore")


# %% pseudo residuals


for (zone, model), df in resids.groupby(level=(0, 1)):
    print(zone, model)
    figs, _ = nplt.diagnostic_plots(
        df["Normal"], df.index.get_level_values(2), save_path=save_path / "Figures" / "Residuals" / f"{zone} - {model}"
    )
    figs[0].suptitle(f"{zone} - {model}")

for model, df in resids.groupby(level=(1)):
    print(model)
    figs, _ = nplt.diagnostic_plots(
        df["Normal"], df.index.get_level_values(2), save_path=save_path / "Figures" / "Residuals" / f"{model}"
    )
    figs[0].suptitle(model)


# %% scores


# geometric mean
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
        save_path / "Tables" / "Basis scores.tex",
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        position="h",
        position_float="centering",
        multicol_align="r",
        multirow_align="r",
        caption=(
            "Scores for the estimated marginal distributions. "
            "Performance seem to be similar for the 3 complicated models across zones, with a slight edge to TAQR. "
            "For DK2 there seems to be a more complicated dynamic which the simpler models cannot capture fully.",
            "Scores of the estimated marginal distributions",
        ),
    )
)
# %% close plots
plt.close("all")
