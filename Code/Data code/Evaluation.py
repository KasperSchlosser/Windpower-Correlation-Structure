import pathlib
import tomllib


import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import numba as nb

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data"
save_path = PATH / "Data" / "Evaluation"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]

rng = np.random.default_rng(42)


@nb.jit(nb.float64[:, :](nb.float64[:, :], nb.float64), nopython=True, nogil=True, parallel=True)
def vars_e(arr, p=0.5):

    n, k = arr.shape
    res = np.zeros((n, n))

    for col in range(k):
        for i in range(n):
            for j in range(n):
                res[i, j] += abs(arr[i, col] - arr[j, col]) ** p

    return res / k


@nb.jit(nb.float64[:, :, :](nb.float64[:, :], nb.float64[:, :], nb.float64), nopython=True, nogil=True, parallel=True)
def VarS(arr, mu, p=0.5):

    n, k = arr.shape

    # mu = vars_e(arr, p)
    res = np.zeros((k, n, n))

    for col in range(k):
        for i in range(n):
            for j in range(n):
                res[col, i, j] += (abs(arr[i, col] - arr[j, col]) ** p - mu[i, j]) ** 2

    return res


# %%

mod = sm.tsa.SARIMAX([0] * 24, order=(2, 0, 0))
params = np.array([0.8, -0.64, 1])
params_wrong = np.array([0.8, 0, 1])
offset = 2
ps = [0.5, 1, 2]
quantiles = [0.5, 0.05, 0.95]

sims_correct = mod.simulate(params, 24, initial_state=[offset] + [0] * 1, repetitions=10000).squeeze()
sims_wrong = mod.simulate(params_wrong, 24, initial_state=[offset] + [0] * 1, repetitions=10000).squeeze()

process = pd.concat(
    [
        pd.DataFrame(np.quantile(sims_correct, quantiles, axis=1).T, columns=[f"{x:.2f}" for x in quantiles]),
        pd.DataFrame(np.quantile(sims_wrong, quantiles, axis=1).T, columns=[f"{x:.2f}" for x in quantiles]),
    ],
    keys=["Correct model", "Incorrect model"],
)
diffs = pd.concat([pd.DataFrame(np.abs(sims_correct - sims_correct[0, :]).T ** p) for p in ps], keys=ps)

ediff = np.stack([vars_e(sims_correct, 0.5), vars_e(sims_wrong, 0.5)])

scores_correct = []
scores_wrong = []

for p in ps:
    e_correct = vars_e(sims_correct, p)
    e_wrong = vars_e(sims_wrong, p)
    scores_correct.append(pd.Series(VarS(sims_correct, e_correct, p).mean(axis=(1, 2))) ** (1 / (2 * p)))
    scores_wrong.append(pd.Series(VarS(sims_correct, e_wrong, p).mean(axis=(1, 2))) ** (1 / (2 * p)))

scores_correct = pd.concat(scores_correct, keys=[str(x) for x in ps], axis=1)
scores_wrong = pd.concat(scores_wrong, keys=[str(x) for x in ps], axis=1)

scores = pd.concat([scores_correct, scores_wrong], keys=["Correct model", "Incorrect model"])


# %%
process.to_pickle(save_path / "process.pkl")
process.to_csv(save_path / "process.csv")

diffs.to_pickle(save_path / "diffs.pkl")
diffs.to_csv(save_path / "diffs.csv")

np.save(save_path / "ediff", ediff)

scores.to_pickle(save_path / "scores.pkl")
scores.to_csv(save_path / "scores.csv")
