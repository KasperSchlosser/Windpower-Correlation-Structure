import pathlib
import tomllib

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import nabqra

from pandas import IndexSlice as idx
from types import SimpleNamespace


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data"
save_path = PATH / "Data"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]


def create_models(quantiles, lims):
    return {
        "Linear": nabqra.quantiles.linear_model(quantiles, *lims, tail_correction=False),
        "Linear-quadratic tail": nabqra.quantiles.linear_model(quantiles, *lims, tail_correction=True),
        "Spline": nabqra.quantiles.spline_model(quantiles, *lims)
    }


# %% exmple dists
plt.close('all')

X = np.linspace(-6.1, 6.1, num=10000)

dists = (
    SimpleNamespace(
        dist=stats.uniform(-6, 12),
        intlim=(-6, 6),
        cdflim=(1e-12, 1-1e-12),
        name="Uniform",
    ),
    SimpleNamespace(
        dist=stats.norm(),
        intlim=(-5, 5),
        cdflim=(1e-12, 1-1e-12),
        name="Normal",
    ),
    SimpleNamespace(
        dist=stats.cauchy(scale=1/6),
        intlim=(-6, 6),
        cdflim=(1e-4, 1-1e-4),
        name="Cauchy",
    ),
    SimpleNamespace(
        dist=stats.beta(4, 17, loc=-6, scale=12),
        intlim=(-6, 2),
        cdflim=(1e-12, 1-1e-12),
        name="Beta",
    ),
)

fig, ax = plt.subplots()

for dist in dists:
    ax.plot(X, dist.dist.pdf(X), label=dist.name)
ax.legend()

fig, ax = plt.subplots()

for dist in dists:
    ax.plot(X, dist.dist.cdf(X), label=dist.name)
ax.legend()


# %%

plt.close('all')
X = np.linspace(-6.1, 6.1, num=10000)
quantiles = np.arange(0.25, 1, 0.25)
# quantiles = np.arange(0.02, 1, 0.02)
model_names = list(create_models([0, 1], [0, 1]).keys())


scores = pd.DataFrame(index=model_names,
                      columns=pd.MultiIndex.from_product([(x.name for x in dists),
                                                          ["W_1", "W_2", "D_{KL}"]]
                                                         )
                      )
estimates = pd.DataFrame(index=X,
                         columns=pd.MultiIndex.from_product([["True"] + model_names,
                                                             (x.name for x in dists),
                                                             ["cdf", "pdf"]]
                                                            )
                         )

for dist in dists:
    print(dist.name)
    label = dist.name
    models = create_models(quantiles, [-6, 6])

    quantile_vals = dist.dist.ppf(quantiles)

    estimates["True", label, "cdf"] = dist.dist.cdf(X)
    estimates["True", label, "pdf"] = dist.dist.pdf(X)

    for name, model in models.items():
        print(name)

        model.fit(quantile_vals)

        estimates[name, label, "cdf"] = model.cdf(X)
        estimates[name, label, "pdf"] = model.pdf(X)

        scores.loc[name, idx[label, "W_1"]] = nabqra.scoring.continous_wasserstein(model.quantile,
                                                                                   dist.dist.ppf,
                                                                                   dist.cdflim,
                                                                                   order=1,
                                                                                   limit=200,
                                                                                   )[0]
        scores.loc[name, idx[label, "W_2"]] = nabqra.scoring.continous_wasserstein(model.quantile,
                                                                                   dist.dist.ppf,
                                                                                   dist.cdflim,
                                                                                   limit=200,
                                                                                   order=2
                                                                                   )[0]
        scores.loc[name, idx[label, "D_{KL}"]] = nabqra.scoring.continous_kl_divergence(model.pdf,
                                                                                        dist.dist.pdf,
                                                                                        dist.intlim,
                                                                                        limit=300,
                                                                                        points=quantiles
                                                                                        )[0]

for dist in dists:
    label = dist.name
    fig, ax = plt.subplots()

    ax.plot(X, estimates["True", label, "cdf"], color="black")
    ax.plot(X, estimates.xs((label, "cdf"), axis=1, level=(1, 2)).iloc[:, 1:], alpha=0.9)
    ax.legend(["True"] + list(models.keys()))

    fig, ax = plt.subplots()

    ax.plot(X, estimates["True", label, "pdf"], color="black")
    ax.plot(X, estimates.xs((label, "pdf"), axis=1, level=(1, 2)).iloc[:, 1:], alpha=0.9)
    ax.legend(["True"] + list(models.keys()))

# %% tail problem
ar1 = 0.85
sigma = 0.7
window = 24
Nwindow = 12

sample_path = np.zeros(window*Nwindow)
pred_path = np.zeros((Nwindow, window))
resid = stats.norm(scale=sigma).rvs(len(sample_path), random_state=360)

for i in range(1, sample_path.shape[0]):
    sample_path[i] = ar1*sample_path[i-1] + resid[i]

pred_path[:, 0] = sample_path[::window]
pred_var = np.zeros(window)
pred_interval = np.zeros((2, window))

for i in range(1, window):
    pred_path[:, i] = ar1*pred_path[:, i-1]
    pred_var[i] = ar1**2*pred_var[i-1] + sigma**2

pred_path = pred_path.ravel()[:, np.newaxis]
pred_var = np.tile(pred_var, Nwindow)
pred_interval = np.outer(np.sqrt(pred_var), stats.norm.ppf([0.05, 0.95]))

pred = np.concat((pred_path, pred_interval + pred_path), axis=1)

dist = stats.norm()
quantiles = np.arange(0.1, 1, 0.1)
quantiles = np.pad(quantiles, (1, 1), constant_values=(0.01, 0.99))

q_vals = dist.ppf(quantiles)
models = create_models(quantiles, [-10, 10])

df = pd.DataFrame(index=range(pred.shape[0]),
                  columns=pd.MultiIndex.from_product((["True"] + list(models.keys()),
                                                      ["Mean", "Lower", "Upper"])))
df["True"] = dist.ppf(stats.norm().cdf(pred))

for name, model in models.items():
    df[name] = model.back_transform(np.broadcast_to(q_vals, (len(pred), len(q_vals))), pred)[0]

fig, ax = nabqra.plotting.band_plot(np.arange(len(df)),
                                    *df["True"].values.T, band_label="True estimates")
ax.plot(np.arange(len(df)),
        dist.ppf(stats.norm().cdf(sample_path)),
        color="black", label="Sample path")

# fig, ax = plt.subplots()

for name, model in models.items():
    l = ax.plot(df[name, "Mean"], label=name)
    ax.plot(df[name, "Lower"], color=l[0].get_color(), linestyle="--")
    ax.plot(df[name, "Upper"], color=l[0].get_color(), linestyle="--")

ax.legend()

# %%
