import pathlib
import tomllib

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import nabqra.quantiles as qm
import nabqra.scoring as scoring

from pandas import IndexSlice as idx

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" 
save_path = PATH / "Data" 

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)

#%% fitting 
dist = stats.norm()
lims = [-6.,6.]
qs = np.pad(np.arange(0.1, 1, 0.1), 1, constant_values = (0.01, 0.99))
q_vals = dist.ppf(qs)



const = qm.constant_model(qs, *lims).fit(q_vals)
linear_notail = qm.linear_model(qs, *lims).fit(q_vals)
linear_tail = qm.linear_model(qs, *lims, tail_correction=True).fit(q_vals)
spline = qm.spline_model(qs, *lims).fit(q_vals)

X = np.linspace(*lims, num = 10000)
X_inv = np.linspace(1e-10,1 - 1e-10, num = 10000)

plt.figure()
plt.plot(X, np.array([const.cdf(X), linear_notail.cdf(X), linear_tail.cdf(X), spline.cdf(X)]).T)
plt.plot(X, dist.cdf(X), color = "black")
plt.legend(["const", "notail", "tail", "spline", "True"])

plt.figure()
plt.plot(X, np.array([linear_notail.pdf(X), linear_tail.pdf(X), spline.pdf(X)]).T)
plt.plot(X, dist.pdf(X), color = "black")
plt.legend(["notail", "tail", "spline", "True"])

plt.figure()
plt.plot(X_inv, np.array([linear_notail.quantile(X_inv), linear_tail.quantile(X_inv), spline.quantile(X_inv)]).T)
plt.plot(X_inv, dist.ppf(X_inv), color = "black")
plt.legend(["notail", "tail", "spline", "True"])