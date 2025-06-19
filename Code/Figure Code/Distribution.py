import pathlib
import tomllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nabqra

from pandas import IndexSlice as idx

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Distribution"
save_path = PATH / "Results" / "Distribution"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]

# %% load data

estimate_df = pd.read_pickle(load_path / "estimates.pkl")
score_df = pd.read_pickle(load_path / "scores.pkl").T
tail_df = pd.read_pickle(load_path / "tail_problem.pkl")

dists = estimate_df.columns.unique(1)
models = estimate_df.columns.unique(0)
X = estimate_df.index

# %% dist plots
fig, ax = plt.subplots()

ax.plot(estimate_df.index, estimate_df.loc[:, idx["True", :, "cdf"]])
ax.legend(dists)
ax.set_xlabel("$x$")
ax.set_ylabel("$P(X<x)$")

fig.savefig(save_path / "Figures" / "Example cdf")
plt.close(fig)

fig, ax = plt.subplots()

ax.plot(estimate_df.index, estimate_df.loc[:, idx["True", :, "pdf"]])
ax.legend(dists)
ax.set_xlabel("$x$")
ax.set_ylabel("$p(x)$")

fig.savefig(save_path / "Figures" / "Example pdf")
plt.close(fig)

for dist in dists:

    fig, ax = plt.subplots()

    ax.plot(X, estimate_df["True", dist, "cdf"], color="black")
    ax.plot(X, estimate_df.xs((dist, "cdf"), axis=1, level=(1, 2)).iloc[:, 1:], alpha=0.9)
    ax.legend(models)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$P(X < x)$")

    fig.savefig(save_path / "Figures" / (dist + " cdf"))
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.plot(X, estimate_df["True", dist, "pdf"], color="black")
    ax.plot(X, estimate_df.xs((dist, "pdf"), axis=1, level=(1, 2)).iloc[:, 1:], alpha=0.9)
    ax.legend(models)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")

    fig.savefig(save_path / "Figures" / (dist + " pdf"))
    plt.close(fig)

# %% score table

score_df.style.format(precision=2).highlight_min(axis=1, props="font-weight:bold;").to_latex(
    save_path / "Tables" / "scores.tex",
    hrules=True,
    clines="skip-last;data",
    convert_css=True,
    position="h",
    position_float="centering",
    multicol_align="r",
    multirow_align="r",
    caption=(
        "The Wasserstein distance (with \( p = 1 \) and \( p = 2 \)) and \gls{kl} for the three models. "
        "The spline model consistently shows the lowest score",
        "Model Scores",
    ),
)


# %% tail

fig, ax = nabqra.plotting.band_plot(np.arange(len(tail_df)), *tail_df["True"].values.T, band_label="True interval")

for name in list(models)[1:]:
    l = ax.plot(tail_df[name, "Mean"], label=name)
    ax.plot(tail_df[name, "Lower"], color=l[0].get_color(), linestyle="--")
    ax.plot(tail_df[name, "Upper"], color=l[0].get_color(), linestyle="--")

ax.legend()
ax.set_xlabel("t")

fig.savefig(save_path / "Figures" / "Tail Problem")
plt.close(fig)
