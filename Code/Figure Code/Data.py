import pathlib
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Data"
save_path = PATH / "Results" / "Distribution"


raw_obs = pd.read_pickle(load_path / "raw_observations.pkl")
raw_ens = pd.read_pickle(load_path / "raw_ensembles.pkl")

clean_obs = pd.read_pickle(load_path / "cleaned_observations.pkl")
clean_ens = pd.read_pickle(load_path / "cleaned_ensembles.pkl")

zones = raw_obs.columns.unique(level = 0)
index = raw_obs.index

#%% illustrate ensembles
# note these difficult to draw


fig, axes = plt.subplots(2,2, sharex = True, sharey = False, squeeze = True)
axes = axes.ravel()

for i, zone in enumerate(zones):
    # mute the ensembles to avoid spaghetti
    axes[i].plot(raw_ens.index, raw_ens[zone].iloc(axis = 1)[1:],
             color = "grey", alpha = 0.3)
    # highlight the HPE
    axes[i].plot(raw_ens.index, raw_ens[zone, "HPE"])

fig.supxlabel("Date")
fig.supylabel("Power production (kW)")

