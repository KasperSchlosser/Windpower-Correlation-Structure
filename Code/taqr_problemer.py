# %%
import scipy.stats as stats
import matplotlib.pyplot as plt
from nabqra.taqr import run_taqr_wrong
import numpy as np
import pandas as pd

# problems
# need more than 1 input variable. crash otherwise
# crash if only 1 quantile (actuals_output[1])
# cannot change bin_size

import pathlib

PATH = pathlib.Path.cwd().parents[0]

save_path = PATH / "Results" / "Basis"


# %% alignment issue

N = 32
sigma = 0.1
loc = -0.5

X = np.tile([0, 0, 1, 1], N // 4)[:, np.newaxis]

X = np.concat((np.ones((N, 1)), X), axis=1)
y = X[:, 1] + sigma * stats.norm().rvs(N, random_state=32) + loc

ix = np.arange(len(y))

X = pd.DataFrame(X, columns=["const", "var"], index=ix)
y = pd.DataFrame(y, columns=["observation"], index=ix)

init = 10
est_quant, taqr_obs, beta = run_taqr_wrong(X, y, [0.1, 0.5, 0.9], init, N, init)

save_idx = ix[init + 1 : N - 1]
taqr_obs = pd.Series(taqr_obs.flatten(), index=save_idx)
taqr_res = pd.DataFrame(np.array(est_quant).T, index=save_idx)

fig, ax = plt.subplots()
plt.scatter(y.index, y, color="C0", label="True obs")


l = plt.plot(X.index, X.iloc[:, -1] + loc, label="True Estimate", color="C0")
plt.fill_between(
    X.index,
    X.iloc[:, -1] + loc + stats.norm(scale=sigma).ppf(0.1),
    X.iloc[:, -1] + loc + stats.norm(scale=sigma).ppf(0.9),
    color=l[0].get_color(),
    alpha=0.3,
)

plt.scatter(taqr_obs.index, taqr_obs, color="C1", marker="x", label="TAQR obs")
l = plt.plot(taqr_res.index, taqr_res.iloc[:, 1], label="TAQR Estimates", color="C1")
plt.fill_between(taqr_res.index, taqr_res.iloc[:, 0], taqr_res.iloc[:, 2], color=l[0].get_color(), alpha=0.3)

plt.legend()

plt.savefig(save_path / "Figures" / "taqr alignment")
plt.close(fig)


# %% init influence

N = 5000
scale = 1

X = stats.norm(scale=1).rvs((N, 1), random_state=12)
X = np.concat((np.ones((N, 1)), X), axis=1)

beta = stats.norm(scale=1).rvs(N, random_state=12).cumsum()[:, np.newaxis]
beta = np.concat((beta, 0.0001 * np.ones((N, 1))), axis=1)

y = np.vecdot(X, beta) + stats.norm(scale=scale).rvs(N, random_state=42)
true = np.vecdot(X, beta)
ix = np.arange(len(y))


fig1, ax = plt.subplots()
fig2, ax2 = plt.subplots()

l = ax.plot(ix, true, label="True")
ax.fill_between(
    ix, true + stats.norm().ppf(0.1) * scale, true + stats.norm().ppf(0.9) * scale, color=l[0].get_color(), alpha=0.3
)
ax2.plot(ix, beta[:, 0], label="True")

for init in [3, 500, 1000, 2500, 4000]:
    est_quant, _, est_beta = run_taqr_wrong(X, y, [0.1, 0.5, 0.9], init, N, init)

    l = ax.plot(ix[init + 2 :], est_quant[1], label=f"initialization: {init} observations")

    ax2.plot(ix[init + 1 :], est_beta[1][:, 0], label=f"initialization: {init} observations")

ax.legend()
# ax.set_title("50% Quantile")
ax2.legend()
# ax2.set_title("$\Beta$")

fig1.savefig(save_path / "Figures" / "median initialiser")
fig2.savefig(save_path / "Figures" / "beta initialiser")

plt.close(fig1)
plt.close(fig2)
