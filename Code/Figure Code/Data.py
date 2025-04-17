import pathlib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Data"
save_path = PATH / "Results" / "Data"


raw_obs = pd.read_pickle(load_path / "raw_observations.pkl")
raw_ens = pd.read_pickle(load_path / "raw_ensembles.pkl")

clean_obs = pd.read_pickle(load_path / "cleaned_observations.pkl")
clean_ens = pd.read_pickle(load_path / "cleaned_ensembles.pkl")

zones = raw_obs.columns.unique(level = 0)
index = raw_obs.index

#%% illustrate ensembles
# this takes a long time to run
# maybe it should be split into individual figures?

fig, axes = plt.subplots(2,2, sharex = True, sharey = False, squeeze = True)
axes = axes.ravel()

for zone, color, ax in zip(zones, plt.rcParams['axes.prop_cycle'].by_key()["color"], axes):
    # mute the ensembles to avoid spaghetti
    ax.plot(raw_ens.index, raw_ens[zone].iloc(axis = 1)[1:],
             color = "grey", alpha = 0.3)
    # highlight the HPE
    ax.plot(raw_ens.index, raw_ens[zone, "HPE"], color= color)
    ax.set_title(f'{zone}')
    
    ax.tick_params("x", rotation = 15)

fig.supxlabel("Date")
fig.supylabel("Power production (kW)")
fig.suptitle("Ensembles")

#save full
fig.savefig(save_path / "Figures" / "Ensembles full")

#Smaller view
axes[0].set_xlim([np.datetime64("2022-08-29"), np.datetime64("2022-09-12")])
fig.savefig(save_path / "Figures" / "Ensembles small")


#%% illustrate Observations

fig, ax = plt.subplots()
# remove original y axis
ax.spines["left"].set_visible(False)
ax.set_yticks([])

ax.tick_params("x", rotation = 15)

lines = []
lims = {
    "DK1-offshore": [0,1250],
    "DK1-onshore": [0,3600],
    "DK2-offshore": [0,1000],
    "DK2-onshore": [0,650],
    }

for zone, color, offset in zip(zones, plt.rcParams['axes.prop_cycle'].by_key()["color"], [0, -0.05,-0.1,-0.15] ):

    twin = ax.twinx()
    p = twin.plot(raw_obs.index, raw_obs[zone], color = color, alpha = 1, label = zone)
    lines.append(p[0])

    twin.spines["left"].set_position(("axes", offset))
    twin.spines["left"].set_visible(True)
    twin.spines["left"].set_color(color)
    twin.yaxis.set_label_position('left')
    twin.yaxis.set_ticks_position('left')
    twin.tick_params(axis = "y", which = "both", colors = color)
    
    twin.set_ylim(*lims[zone])
    
    twin.set_ylabel(zone, labelpad = -1, color = color)
    
fig.savefig(save_path / "Figures" / "Observations full")
ax.set_xlim([np.datetime64("2022-08-29"), np.datetime64("2022-09-12")])
fig.savefig(save_path / "Figures" / "Observations small")

#%% comparison ensembles, observations
fig, axes = plt.subplots(2,2, sharex = True, sharey = False, squeeze = True)
axes = axes.ravel()

for zone, color, ax in zip(zones, plt.rcParams['axes.prop_cycle'].by_key()["color"], axes):
    
    ax.plot(clean_ens.index, clean_ens[zone].median(axis = 1), color= color)
    ax.fill_between(clean_ens.index, clean_ens[zone].min(axis = 1), clean_ens[zone].max(axis = 1),
                    color= color, alpha = 0.3)
    ax.scatter(clean_obs.index, clean_obs[zone], color = "black", s = 1)
    ax.set_title(f'{zone}')
    
    ax.tick_params("x", rotation = 15)

fig.supxlabel("Date")
fig.supylabel("Power production (kW)")
fig.suptitle("Ensembles vs Observations")
axes[0].set_xlim([np.datetime64("2022-09-01"), np.datetime64("2022-09-14")])

fig.savefig(save_path / "Figures" / "Comparison")
