import pathlib
import tomllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import IndexSlice as idx

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Basis"
save_path = PATH / "Data" / "Basis"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]

plt.close("all")
# %%

history = pd.read_pickle(load_path / "History.pkl")
preds = pd.read_pickle(load_path / "Basis Quantiles.pkl")
models = history.index.unique(1)


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
axes = axes.ravel()
for ax, zone in zip(axes, zones):
    ax.plot(history.loc[zone, "loss"].reset_index().pivot(index="level_1", columns="Model"))
    ax.legend(models[[1, 0, 2]])
    ax.set_title(zone)
fig.suptitle("Train Loss")


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
axes = axes.ravel()
for ax, zone in zip(axes, zones):
    ax.plot(history.loc[zone, "val_loss"].reset_index().pivot(index="level_1", columns="Model"))
    ax.legend(models[[1, 0, 2]])
    ax.set_title(zone)
fig.suptitle("Validation Loss")

# %%
for model in models:
    fig, ax = plt.subplots()
    ax.set_title(model)
    for zone in zones:
        l = ax.plot(history.loc[zone, model]["loss"], label=zone)
        ax.plot(history.loc[zone, model]["val_loss"], color=l[0].get_color(), linestyle="--")
    ax.legend()
