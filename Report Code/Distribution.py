import pathlib
import os
os.chdir(pathlib.Path("..").resolve())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import Code.quantiles as qm
import Code.evaluation as evaluation

from pandas import IndexSlice as idx

#%%
#load data in
#needs to be made to do all 4 data sets, when they are done

PATH = pathlib.Path()

#for moth of this just use DK2 onshore, seems to be the most stabel
zone = "DK2-onshore"
 

data = pd.read_pickle(PATH / "Data" / "NABQR_results_full.pkl")
quantiles = [float(x) for x in data["DK1-offshore", "Quantiles"].columns]

#tail data
data_tails = pd.read_pickle(PATH / "Data" / "NABQR Results" / "tails.pkl")
quantiles_tail = data_tails["Quantiles"].columns


#%% tails of qr

#not much difference between 99% and 99.99%
#values below black line indicate numerical problems
fig, ax = plt.subplots(figsize = (14,8), layout = "tight")
sns.scatterplot(x = data_tails["Quantiles"].iloc[:,0], y = data_tails["Quantiles"].iloc[:,-2], ax = ax)
ax.set_xlabel("99% Quantile")
ax.set_ylabel("99.99% Quantile")
ax.axline((0,0), slope = 1, color = 'black')
ax.axline((0,0), slope = 1.2, color = 'crimson')

#numerical problems can be seend
fig, ax = plt.subplots(figsize = (14,8), layout = "tight")
ax.plot(data_tails.index, data_tails["Quantiles"].iloc[:,::10])
ax.legend(quantiles_tail[::10])

ax.set_xlim([np.datetime64("2024-07-26T12:00"),np.datetime64("2024-07-28T06:00")] )
ax.set_ylim([200,500])

fig.savefig(PATH / "Figures" / "Quantile fitting" / "crossing1.png")

ax.set_xlim([np.datetime64("2024-08-21T08:00"),np.datetime64("2024-08-23T02:00")] )
ax.set_ylim([1050,1410])

fig.savefig(PATH / "Figures" / "Quantile fitting" / "crossing2.png")

#%% Constant estimator
# very easy and fast
# discrete - can be nice

const_model = qm.constant_model(quantiles)
_, const_pseudo = const_model.transform(data[zone, "Quantiles"].values, data[zone,"Observed"].values.squeeze())

evaluation.evaluate_pseudoresids(const_pseudo.squeeze(),
                           save_path = PATH / "Figures" / "Quantile fitting",
                           name = "Constant",
                           close_figs  = True)


#%% Piecewise linear
# easy calculation 

max_val = 1000
min_val = -2
pwl = qm.piecewise_linear_model(quantiles, min_val, max_val)
_, pwl_pseudo = pwl.transform(data[zone, "Quantiles"].values, data[zone, "Observed"].values)

evaluation.evaluate_pseudoresids(pwl_pseudo.squeeze(),
                           save_path = PATH / "Figures" / "Quantile fitting",
                           name = "pwl",
                           close_figs  = True)


#%% problems with piecewise linear

#what should max values be
q_max = data[zone, "Quantiles", '0.99'].max()
d_max = data[zone, "Observed","Observed"].max()
arrowprops = {"facecolor":'navy'}


fig, ax = plt.subplots(figsize = (14,8), layout = "tight")
sns.scatterplot(x = data[zone, "Quantiles", '0.99'], y = data[zone, "Observed", "Observed"], ax = ax)
ax.axline((0,0), slope = 1, color = 'navy', linestyle = "dashed")
ax.axline((0,0), slope = 1.5, color = 'crimson', linestyle = "dashdot")
ax.axline((0,d_max), slope = 0, color = 'black')
ax.axline((0,q_max), slope = 0, color = 'black')

ax.annotate("Observation = 99% quantile",
             (750,750), (550,900),
             color = 'navy',
             arrowprops = {"color":'navy', "arrowstyle":"->"},
)

ax.annotate("Observation = 1.5 * 99% quantile",
             (500,750), (200,900),
             color = 'crimson',
             arrowprops = {"color":'crimson', "arrowstyle":"->"},
)

ax.annotate("Max Observation",
             (0,d_max), (0, d_max - 100),
             color = 'black',
             arrowprops = {"color":'black', "arrowstyle":"->"},
)

ax.annotate("max 99% quantile",
             (0,q_max), (0, q_max - 100),
             color = 'black',
             arrowprops = {"color":'black', "arrowstyle":"->"},
)

ax.set_xlabel("99% quantile")
ax.set_ylabel("observed value")

fig.savefig(PATH / "Figures" / "Quantile fitting" / "max_problem.png")

#%% spike problem
# The max/min value can be far from the actual distribution



