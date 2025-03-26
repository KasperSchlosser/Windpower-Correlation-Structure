import pathlib
import os
os.chdir(pathlib.Path.cwd().parents[1].resolve())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import Code.evaluation as evaluation

from pandas import IndexSlice as idx


#%% load data
#lets crash

PATH = pathlib.Path()
load_path = PATH / "Data" / "Autocorrelation"
figure_path = PATH / "Figures" / "Autocorrelation"
table_path = PATH / "Tables" / "Autocorrelation"

observations = pd.read_pickle(load_path / "observations.pkl")
forecasts = pd.read_pickle(load_path / "forecasts.pkl")
scores = pd.read_pickle(load_path / "scores.pkl")
simulations = pd.read_pickle(load_path / "simulations.pkl")

zones = observations.columns.get_level_values(0).unique()

burn_in = 240

#%% forecasts in orignal space


plots = [
    {
     "zone": "DK1-onshore",
     "space": "original",
     "xlims": (np.datetime64("2024-08-05"), np.datetime64("2024-08-14")),
     "ylims": (-100, 3100)
    },
    {
     "zone": "DK1-onshore",
     "space": "cdf",
     "xlims": (np.datetime64("2024-08-05"), np.datetime64("2024-08-14")),
     "ylims": (0,1)
    },
    {
     "zone": "DK1-onshore",
     "space": "normal",
     "xlims": (np.datetime64("2024-08-05"), np.datetime64("2024-08-14")),
     "ylims": (-3,3)
    },
    {
     "zone": "DK2-offshore",
     "space": "original",
     "xlims": (np.datetime64("2024-08-08"), np.datetime64("2024-08-18")),
     "ylims": (-100, 1100)
    }
]

def line_plots(zone, space, xlims, ylims, N_step = 24, save = True, Name = None, plot_variance = True):
    # function for making this specific plot
    # not general, just saves a lot of writing
    x = forecasts.index
    
    fig, ax = plt.subplots(figsize = (14,8), layout = "tight")
    sns.scatterplot(x = x, y = observations[zone, space], color = 'black', marker = 'x', ax = ax, label = "acutal")
    
    for color, model in zip(["crimson", "olivedrab", "navy"], ["SARMA", "ARMA", "nabqr"]):
        sns.lineplot(x = x, y = forecasts[zone, model, space, "estimate"], color = color, ax = ax, label = model)
        ax.fill_between(
            x,
            forecasts[zone, model, space, "lower prediction"],
            forecasts[zone, model, space, "upper prediction"],
            color = color,
            alpha = 0.3
        )

    ax.vlines(
        x[N_step-1::N_step],
        ylims[0],
        ylims[1],
        color = 'black',
        linestyle = '--'
    )
    ax.set_title(zone)
    ax.legend(loc = "upper right")
    ax.set_ylabel("Production")
    ax.set_xlabel("Date")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    if save:
        if Name is None:  Name = ("-".join([zone, space, "forecast"]))
        fig.savefig( (figure_path / Name).with_suffix(".png") )
        
    return fig, ax


for kwargs in plots:
    line_plots(**kwargs)

#%% interval plots

plots = [
    {
     "zone": "DK1-onshore",
     "space": "original",
     "xlims": (np.datetime64("2024-08-05"), np.datetime64("2024-08-14")),
     "ylims": (-1200, 1800)
    },
    {
     "zone": "DK1-onshore",
     "space": "cdf",
     "xlims": (np.datetime64("2024-08-05"), np.datetime64("2024-08-14")),
     "ylims": (-1,1)
    },
    {
     "zone": "DK1-onshore",
     "space": "normal",
     "xlims": (np.datetime64("2024-08-05"), np.datetime64("2024-08-14")),
     "ylims": (-3,3)
    },
]

def var_plots(zone, space, xlims, ylims, N_step = 24, save = True, Name = None):
    # function for making this specific plot
    # not general, just saves a lot of writing
    x = forecasts.index
    y = forecasts.xs(space, level = 2, axis = 1)[zone]
    
    fig, ax = plt.subplots(figsize = (14,8), layout = "tight")
    
    
    for color, model in zip(["crimson", "olivedrab", "navy"], ["SARMA", "ARMA", "nabqr"]):
    
        #sns.scatterplot(x = x, y = df_resids[zone, space] - y[model,"estimate"], color = color, marker = 'x', ax = ax)
        sns.lineplot( x = x, y = y[model, "lower prediction"] - y[model,"estimate"], color = color, linestyle = "--", ax = ax, label = model)
        sns.lineplot( x = x, y = y[model, "upper prediction"] - y[model,"estimate"], color = color, linestyle = "--", ax = ax)
    
    ax.vlines(
        x[N_step-1::N_step],
        ylims[0],
        ylims[1],
        color = 'black',
        linestyle = '--'
    )
    ax.axline((0,0), slope = 0, color = "black")
    ax.set_title(zone)
    ax.legend(loc = "upper right")
    ax.set_ylabel("Production interval")
    ax.set_xlabel("Date")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    if save:
        if Name is None:  Name = ("-".join([zone, space, "interval"]))
        fig.savefig( (PATH / "Figures" / "Autocorrelation" / Name).with_suffix(".png") )
        
    return fig, ax

for kwargs in plots:
    var_plots(**kwargs)


#%% actual by predicted
"""
def abp_plot(zone, space, xlim, ylim, save = True, Name = None):
    # function for making this specific plot
    # not general, just saves a lot of writing

    fig, ax = plt.subplots(figsize = (14,8), layout = "tight")
    
    sns.scatterplot(
        x = df_forecast[zone, "SARMA", space, "estimate"],
        y = df_resids[zone, space],
        color = 'crimson',
        ax = ax
    )
    sns.scatterplot(
        x = df_forecast[zone, "ARMA", space, "estimate"],
        y = df_resids[zone, space],
        color = 'olivedrab',
        ax = ax
    )
    sns.scatterplot(
        x = df_forecast[zone, "nabqr", space, "estimate"],
        y = df_resids[zone, space],
        color = 'navy',
        ax = ax
    )
    ax.axline((0,0), slope = 1, color = 'black')
    ax.set_title(zone)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(["SARMA", "ARMA", "NABQR"])

    if save:
        if Name is None:  Name = ("-".join([zone, space, "actual_predicted"]))
        fig.savefig( (PATH / "Figures" / "Correlation Structure" / Name).with_suffix(".png") )
        
    return fig, ax


xlims = [
    (-10,1400),
    (-40,3700),
    (-10,1100),
    (-55,550)
]

ylims = [
    (-10,1200),
    (-10,3400),
    (-10,1000),
    (-10,580)
]
for zone, xlim, ylim in zip(zones, xlims, ylims):
    abp_plot(zone, "original", xlim, ylim)
    abp_plot(zone, "cdf", (0,1), (0,1))
    abp_plot(zone, "normal", (-3,3), (-3,3))
    
"""
#%% scores 
tmp = scores.T.reset_index(names = ["zone", "metric"]).melt(value_vars = ["nabqr", "SARMA", "ARMA"],
                                                               var_name = "model",
                                                               value_name ="score",
                                                               id_vars = ["zone","metric"])
fig = sns.catplot(tmp,
                x = "model", y = "score", hue = "model",
                row = "metric", col = "zone", kind = "bar",
                palette = {"nabqr": "navy", "SARMA":"crimson", "ARMA":"olivedrab"},
                height = 3, aspect = 1.2,
                sharey = False, sharex=False,
                margin_titles = True,
                dodge = False)
fig.tight_layout()

fig.savefig( (PATH / "Figures" / "Correlation Structure" / "scores.png" ))

scores.T.style\
    .format(precision = 2)\
    .highlight_min(axis = 1, props = "font-weight:bold")\
    .to_latex(table_path / "scores.tex",
              caption = "Model scores for each zone",
              label = "fig:forecasteval:scores",
              hrules = True,
              clines = "skip-last;data",
              position = "h",
              position_float = "centering",
              convert_css = True)


#%% variogram

fig, axes = plt.subplots(len(zones), len(scores.index),
                         figsize = (12,14), layout = "constrained",
                         sharex = True, sharey = True)

c_range = []
for i, zone in enumerate(zones):
    for j, p in enumerate(scores.index):
        
        actuals = observations[zone,"original"].values.squeeze()[burn_in:]
        sim = simulations.loc[:, idx[zone, p, "original", :]].values[burn_in:]
        
        score, variogram = evaluation.variogram_score(sim, actuals)
        #variogram = np.log(variogram)
        np.fill_diagonal(variogram, 0)
        
        if p == "nabqr": c_range.append((variogram.min(),variogram.max()))
        
        
        cbar = False
        if j == len(scores.index)-1: cbar = True
        sns.heatmap(variogram,
                    ax = axes[i,j],
                    vmin = c_range[i][0], vmax = c_range[i][1],
                    xticklabels = 1, yticklabels = 1,
                    cmap = "viridis",
                    cbar = cbar)
        
        if i == 0: axes[i,j].set_title(p)
        if j == 0: axes[i,j].set_ylabel(zone)

fig.savefig( figure_path / "variograms.png" )