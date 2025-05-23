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
models = history.index.unique(1)

plt.close("all")
# %% NN training

fig, axes = plt.subplots(2, 2, sharex=True, sharey=False)
axes = axes.ravel()
for ax, zone in zip(axes, zones):
    ax.semilogy(history.loc[zone, "loss"].unstack("Model"))
    ax.legend(models)
    ax.set_title(zone)
fig.suptitle("Train Loss")
fig.supxlabel("Epoch")
fig.supylabel("loss")


fig, axes = plt.subplots(2, 2, sharex=True, sharey=False)
axes = axes.ravel()
for ax, zone in zip(axes, zones):
    ax.semilogy(history.loc[zone, "val_loss"].unstack("Model"))
    ax.legend(models)
    ax.set_title(zone)
fig.suptitle("Validation Loss")
fig.supxlabel("Epoch")
fig.supylabel("loss")

# %%
for model in models:

    fig, ax = plt.subplots()
    for zone in zones:
        l = ax.semilogy(history.loc[zone, model]["loss"], label=f"{zone} - Train loss")
        ax.semilogy(
            history.loc[zone, model]["val_loss"],
            color=l[0].get_color(),
            linestyle="--",
            label=f"{zone} - Validation loss",
        )
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss")


# %% pseudo residuals


for (zone, model), df in resids.groupby(level=(0, 1)):
    print(zone, model)
    figs, _ = nplt.diagnostic_plots(
        df["Normal"], df.index.get_level_values(2), save_path=save_path / "Figures" / "Residuals" / f"{zone} - {model}"
    )
# %%
plt.close("all")
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
min_loc = scores == min_score
gmap = np.log(scores / min_score)

(
    scores.style.format(precision=2)
    .background_gradient(cmap="Reds", vmin=0, vmax=np.log(2), axis=None, gmap=gmap)
    .apply(
        lambda x: pd.DataFrame(
            np.where(min_loc, "font-weight: bold; font-style: italic", ""),
            index=scores.index,
            columns=scores.columns,
        ),
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
            "Scores for the ensembles and models scores. "
            "Scores are similar across models, except in DK2-onshores where the ensemble forecast is really shit",
            "Scores of the estimated marginal distributions",
        ),
    )
)

print(scores)

# %%


# %%

preds = pd.read_pickle(load_path / "Basis Quantiles.pkl")
