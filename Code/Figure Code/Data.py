import pathlib
import pandas as pd
import numpy as np
import tomllib
import matplotlib.pyplot as plt

import nabqra

from pandas import IndexSlice as idx


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Data"
save_path = PATH / "Results" / "Data"


raw_obs = pd.read_pickle(load_path / "raw_observations.pkl")
raw_ens = pd.read_pickle(load_path / "raw_ensembles.pkl")

clean_obs = pd.read_pickle(load_path / "cleaned_observations.pkl")
clean_ens = pd.read_pickle(load_path / "cleaned_ensembles.pkl")

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)

# %% Illustrate data
period = [np.datetime64("2023-10-17"), np.datetime64("2023-11-02")]

for zone in parameters["Zones"]:

    fig, ax = plt.subplots(figsize=(8, 4))
    index = clean_obs.loc[zone].index

    ax.plot(index, clean_ens.loc[zone, "HPE"], label="HPE")
    ax.scatter(index, clean_obs.loc[zone], label="Observed Production", s=1)

    ax.plot(index, clean_ens.loc[zone], color="black", alpha=0.1)
    ax.plot(index[0], 0, color="black", alpha=0.5, label="Ensembles")

    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Power production (MW)")

    fig.savefig(save_path / "Figures" / f"{zone} full")
    ax.set_xlim(period)
    fig.savefig(save_path / "Figures" / f"{zone} small")

# %% comparison

df = clean_obs.unstack(0)
fig, ax = plt.subplots()

ax.plot(df.index, df)

ax.legend(df.columns)
ax.set_xlabel("Date")
ax.set_ylabel("Power production (MW)")

fig.savefig(save_path / "Figures" / "Comparison full")
ax.set_xlim(period)
fig.savefig(save_path / "Figures" / "Comparison small")

# %% tables
obs = pd.concat((raw_obs, clean_obs), keys=["Raw", "Clean"], axis=1)
ens = pd.concat((raw_ens, clean_ens), keys=["Raw", "Clean"], axis=0).stack()
index = obs.index.unique(-1)


def Negative(x):
    return (x < 0).sum()


def Problematic(x):
    return ((0 < x) & (x < parameters["low_value"])).sum()


def Nan(x):
    return x.isna().sum()


def Inf(x):
    return (x == np.inf).sum() + (x == -np.inf).sum()


obs_table = obs.groupby(level=0).agg(["count", "min", "max", Negative, Problematic, Nan]).stack(level=0)
ens_table = ens.groupby(level=(1, 0)).agg(["count", "min", "max", Nan])


obs_table.style.format(precision=2).to_latex(
    save_path / "Tables" / "Observation Summary.tex",
    hrules=True,
    clines="skip-last;data",
    convert_css=True,
    position="h",
    position_float="centering",
    multicol_align="r",
    multirow_align="r",
    caption=(
        f"Summary table for the observed production data. First datapoint {index.min()} last datapoint {index.max()}. "
        f"Problematic values are values $0 \leq x < {parameters['low_value']}$",
        "Summary table for Production",
    ),
)

ens_table.style.format(precision=2).to_latex(
    save_path / "Tables" / "Ensemble Summary.tex",
    hrules=True,
    clines="skip-last;data",
    convert_css=True,
    position="h",
    position_float="centering",
    multicol_align="r",
    multirow_align="r",
    caption=(
        f"Summary table for the ensemble data. First datapoint {index.min()} last datapoint {index.max()}. ",
        "Summary table for Production",
    ),
)

# %% Illustrate data split

index = clean_obs.index.unique(1)
train_index = index[: parameters["train_size"]]
test_index = index[parameters["train_size"] :]

fig, ax = plt.subplots()
ax.plot(train_index, clean_obs.loc[idx["DK1-onshore", train_index]].values, label="Train")
ax.plot(test_index, clean_obs.loc[idx["DK1-onshore", test_index]].values, label="Test")

ax.set_xlabel("Date")
ax.set_ylabel("Power Production (mW)")
ax.legend()
fig.savefig(save_path / "Figures" / "Datasplit")


# %%
split_table = pd.DataFrame(
    [
        [train_index[0], test_index[0]],
        [train_index[-1], test_index[-1]],
        [len(train_index), len(test_index)],
        [len(train_index) / 24, len(test_index) / 24],
    ],
    index=["Start", "End", "Datapoints", "Datapoints (Days)"],
    columns=["Train", "Test"],
).T
split_table.style.format(precision=2).to_latex(
    save_path / "Tables" / "Datasplit.tex",
    hrules=True,
    clines=None,
    label="data:table:split",
    position="h",
    position_float="centering",
    multicol_align="r",
    multirow_align="r",
    caption=(
        (
            "Data is split into two parts. A train set and a test set. "
            "Split was chosen to be 64\% - 36\%.  Giving roughly 1 year of test data."
        ),
        "Train-test split of data",
    ),
)

plt.close("all")
