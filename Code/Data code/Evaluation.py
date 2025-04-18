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

#%% wasserstein

loc = 1
scale = 2
dist1 = stats.norm()
dist2 = stats.norm(loc = loc, scale = scale)

X = np.linspace(1e-12,1-1e-12, num = 10000)
X2 = np.linspace(-6,6, num = 10000)

plt.close('all')
plt.figure()
plt.plot(X, np.array([dist1.ppf(X), dist2.ppf(X)]).T)
plt.fill_between(X, dist1.ppf(X), dist2.ppf(X), color = 'grey', alpha = 0.3)
plt.ylim([-6,6])
plt.legend(["N(0,1^2)", f'N({loc},{scale}^2)', "Wasserstein Area"])

plt.figure()
plt.plot(X2, np.array([dist1.cdf(X2), dist2.cdf(X2)]).T)
plt.fill_between(X2, dist1.cdf(X2), dist2.cdf(X2), color = 'grey', alpha = 0.3)
plt.legend(["N(0,1^2)", f'N({loc},{scale}^2)', "Wasserstein Area"])


#%%
[scoring.continous_wasserstein(dist1.ppf, dist2.ppf, (0.0001,0.9999), order = p)[0] for p in np.linspace(0.1,3,100)]



#%% kl Divergence
dist1 = stats.norm()
dist2 = stats.t(5)

X = np.linspace(-10,10, num = 10000)

plt.close('all')
plt.figure()

plt.fill_between(X, dist1.pdf(X), alpha = 0.6)
plt.fill_between(X, dist2.pdf(X), alpha = 0.6)

plt.figure()
plt.fill_between(X, dist1.pdf(X) * np.log(dist1.pdf(X) / dist2.pdf(X)))
#plt.fill_between(X, dist2.pdf(X) * np.log(dist2.pdf(X) / dist1.pdf(X)))


#forecast
#MAE, RMSE CRPS, VARS måske interval/wcrps?

