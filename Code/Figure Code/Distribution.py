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
lstm_data = pd.read_pickle(load_path / "lstm_observations.pkl")
taqr_data = pd.read_pickle(load_path / "taqr_observations.pkl")
beta_data = pd.read_pickle(load_path / "beta_observations.pkl")
actuals = pd.read_pickle(load_path / ".." / "NABQR" / "Actuals.pkl")

# currently there are some inf finite predictions, oops need to fix

lstm_data.replace(-np.inf, -6, inplace=True)
taqr_data.replace(-np.inf, -6, inplace=True)

dists = estimate_df.columns.unique(1)
models = estimate_df.columns.unique(0)
X = estimate_df.index
time_index = lstm_data.index


# %% dist plots
fig, ax = plt.subplots()

ax.plot(estimate_df.index, estimate_df.loc[:, idx["True", :, "cdf"]])
ax.legend(dists)
ax.set_xlabel("$x$")
ax.set_ylabel("$P(X<x)$")

fig.savefig(save_path / "Figures" / "example_cdf")
plt.close(fig)

fig, ax = plt.subplots()

ax.plot(estimate_df.index, estimate_df.loc[:, idx["True", :, "pdf"]])
ax.legend(dists)
ax.set_xlabel("$x$")
ax.set_ylabel("$p(x)$")

fig.savefig(save_path / "Figures" / "example_pdf")
plt.close(fig)


for dist in dists:

    fig, ax = plt.subplots()

    ax.plot(X, estimate_df["True", dist, "cdf"], color="black")
    ax.plot(X, estimate_df.xs((dist, "cdf"), axis=1, level=(1, 2)).iloc[:, 1:], alpha=0.9)
    ax.legend(models)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")

    fig.savefig(save_path / "Figures" / (dist + "_cdf"))

    fig, ax = plt.subplots()

    ax.plot(X, estimate_df["True", dist, "pdf"], color="black")
    ax.plot(X, estimate_df.xs((dist, "pdf"), axis=1, level=(1, 2)).iloc[:, 1:], alpha=0.9)
    ax.legend(models)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")

    fig.savefig(save_path / "Figures" / (dist + "_pdf"))

# %% score table

score_df.style.format(precision=2).highlight_min(axis=1, props="font-weight:bold;").to_latex(
    save_path / "Tables" / "scores.tex",
    hrules=True,
    clines="skip-last;index",
    convert_css=True,
    position="h",
    position_float="centering",
    multicol_align="r",
    multirow_align="r",
    column_format="ccccc",
    caption=(
        "The wasserstein distance (for $p = 1$ and $p = 2$) "
        "along with KL-divergence for the 3 models. "
        "Each model evaulated for each of the 4 example distributions "
        "Except for the Beta distribution the spline model always has the lowest score.",
        "Quantile model scores for the example distributions",
    ),
)


# %% tail


fig, ax = nabqra.plotting.band_plot(np.arange(len(tail_df)), *tail_df["True"].values.T, band_label="True estimates")

for name in list(models)[1:]:
    l = ax.plot(tail_df[name, "Mean"], label=name)
    ax.plot(tail_df[name, "Lower"], color=l[0].get_color(), linestyle="--")
    ax.plot(tail_df[name, "Upper"], color=l[0].get_color(), linestyle="--")

ax.legend()
ax.set_xlabel("t")

fig.savefig(save_path / "Figures" / "Tail Problem")

# %% transformed plots

small_lim = [np.datetime64("2024-02-01"), np.datetime64("2024-02-15")]

fig, axes = plt.subplots(2, 2, sharex=True)
axes = axes.ravel()


for ax, zone in zip(axes, zones):

    twin = ax.twinx()
    twin.plot(actuals.index, actuals[zone].values, color="black", linewidth=0.5, label="Observed")
    twin.set_ylabel("Production (kW)", rotation=-90)
    twin.grid(False)

    ax.plot(time_index, lstm_data[zone, "CDF"], label="LSTM", linewidth=1)
    ax.plot(time_index, taqr_data[zone, "CDF"], label="TAQR", linewidth=1)
    ax.plot(time_index, beta_data[zone, "CDF"], label="Beta", linewidth=1)
    ax.legend()
    ax.set_title(zone)
fig.supxlabel("Date")
fig.supylabel("CDF-residual")
axes[2].tick_params(axis="x", rotation=15)
axes[3].tick_params(axis="x", rotation=15)

fig.savefig(save_path / "Figures" / "CDF Residual_full")

ax.set_xlim(small_lim)

fig.savefig(save_path / "Figures" / "CDF Residual_small")


fig, axes = plt.subplots(2, 2, sharex=True)
axes = axes.ravel()

for ax, zone in zip(axes, zones):

    ax.plot(time_index, lstm_data[zone, "Normal"], label="LSTM", linewidth=1)
    ax.plot(time_index, taqr_data[zone, "Normal"], label="TAQR", linewidth=1)
    ax.plot(time_index, beta_data[zone, "Normal"], label="Beta", linewidth=1)
    ax.set_title(zone)
    ax.set_ylim([-3, 3])

    ax.legend()
fig.supxlabel("Date")
fig.supylabel("Normal-residual")
axes[2].tick_params(axis="x", rotation=15)
axes[3].tick_params(axis="x", rotation=15)


fig.savefig(save_path / "Figures" / "Normal Residual_full")

ax.set_xlim(small_lim)

fig.savefig(save_path / "Figures" / "Normal Residual_small")

# %%%
print("lstm")
print(np.sqrt((lstm_data.xs("Normal", level=1, axis=1) ** 2).mean()))
print("taqr")
print(np.sqrt((taqr_data.xs("Normal", level=1, axis=1) ** 2).mean()))
print("beta")
# print(np.sqrt((beta_data.xs("Normal", level = 1, axis = 1)**2).mean()))

# %% residual plots

for zone in zones:
    nabqra.plotting.pseudoresid_diagnostics(
        taqr_data[zone, "Normal"], zone + " TAQR", save_path=save_path / "Figures" / "Residuals" / "taqr"
    )
    nabqra.plotting.pseudoresid_diagnostics(
        lstm_data[zone, "Normal"], zone + " LSTM", save_path=save_path / "Figures" / "Residuals" / "lstm"
    )
