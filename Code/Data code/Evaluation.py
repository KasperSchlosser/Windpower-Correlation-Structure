import pathlib
import tomllib


import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data"
save_path = PATH / "Data" / "Evaluation"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]

rng = np.random.default_rng(42)

# %%
N = 100000
T = 15
N_steps = 100
ar1 = 0.75 ** (1 / N_steps)
sigma = 1 / N_steps
poffset = 1

model = sm.tsa.ARIMA([0, 0], order=(1, 0, 0), trend="n")
X = np.linspace(0, T, num=T * N_steps)


samples = model.simulate([ar1, sigma], T * N_steps, repetitions=N, initial_state=-poffset, random_state=rng).squeeze()
samples = samples + poffset


expected = np.zeros_like(X)
expected[0] = -poffset
var = np.zeros_like(X)
for i in range(1, len(X)):
    expected[i] = ar1 * expected[i - 1]
    var[i] = ar1**2 * var[i - 1] + sigma
expected = expected + poffset

process_interval = stats.norm().ppf(1 - 0.05) * np.sqrt(var)

samples = pd.DataFrame(samples, index=X)
theoretical = pd.DataFrame(
    {"Expected": expected, "Variance": var, "Upper": expected + process_interval, "Lower": expected - process_interval},
    index=X,
)
# %%
samples.to_pickle(save_path / "Samples.pkl")
samples.to_csv(save_path / "Samples.csv")

theoretical.to_pickle(save_path / "Theoretical.pkl")
theoretical.to_csv(save_path / "Theoretical.csv")
