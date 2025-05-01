import pathlib
import tomllib
import pandas as pd
import numpy as np
from scipy import stats
import nabqra
from pandas import IndexSlice as idx
from types import SimpleNamespace

# Define paths
PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data"
save_path = PATH / "Data" / "Distribution"

# Load parameters
with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]

# Function to create models


def create_models(quantiles, lims):
    return {
        "Linear": nabqra.quantiles.linear_model(quantiles, *lims, tail_correction=False),
        "Linear - tail": nabqra.quantiles.linear_model(quantiles, *lims, tail_correction=True),
        "Spline": nabqra.quantiles.spline_model(quantiles, *lims)
    }


# %% Generate example distributions
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

# Create dataframes for scores and estimates
quantiles = np.arange(0.25, 1, 0.25)
model_names = list(create_models([0, 1], [0, 1]).keys())
scores = pd.DataFrame(index=model_names,
                      columns=pd.MultiIndex.from_product([(x.name for x in dists),
                                                          [r"$W_1$", r"$W_2$", r"$D_{KL}$"]]
                                                         )
                      )
estimates = pd.DataFrame(index=X,
                         columns=pd.MultiIndex.from_product([["True"] + model_names,
                                                             (x.name for x in dists),
                                                             ["cdf", "pdf"]]
                                                            )
                         )

# Calculate scores and estimates
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
        scores.loc[name, idx[label, r"$W_1$"]] = nabqra.scoring.continous_wasserstein(model.quantile,
                                                                                      dist.dist.ppf,
                                                                                      dist.cdflim,
                                                                                      order=1,
                                                                                      limit=200,
                                                                                      )[0]
        scores.loc[name, idx[label, r"$W_2$"]] = nabqra.scoring.continous_wasserstein(model.quantile,
                                                                                      dist.dist.ppf,
                                                                                      dist.cdflim,
                                                                                      limit=200,
                                                                                      order=2
                                                                                      )[0]
        scores.loc[name, idx[label, r"$D_{KL}$"]] = nabqra.scoring.continous_kl_divergence(model.pdf,
                                                                                           dist.dist.pdf,
                                                                                           dist.intlim,
                                                                                           limit=300,
                                                                                           points=quantiles
                                                                                           )[0]

# Save scores and estimates to files
scores.to_csv(save_path / "scores.csv")
scores.to_pickle(save_path / "scores.pkl")
estimates.to_csv(save_path / "estimates.csv")
estimates.to_pickle(save_path / "estimates.pkl")

# %% Tail problem data generation
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

df.to_csv(save_path / "tail_problem.csv")
df.to_pickle(save_path / "tail_problem.pkl")

# %% Zone Data

actuals = pd.read_pickle(load_path / "NABQR" / "actuals.pkl")
lstm_estimates = pd.read_pickle(load_path / "NABQR" / "lstm_quantiles.pkl")
taqr_estimates = pd.read_pickle(load_path / "NABQR" / "taqr_quantiles.pkl")


lstm_observations = pd.DataFrame(index=actuals.index,
                                 columns=pd.MultiIndex.from_product((
                                     zones,
                                     ["CDF", "Normal"]
                                 )),
                                 dtype=np.float64
                                 )

taqr_observations = pd.DataFrame(index=actuals.index,
                                 columns=pd.MultiIndex.from_product((
                                     zones,
                                     ["CDF", "Normal"]
                                 )),
                                 dtype=np.float64
                                 )

# i just assume they used the same quantiles
# which i did for now
quantiles = lstm_estimates.columns.unique(1).values.astype(np.float64)
for zone in zones:
    model = nabqra.quantiles.spline_model(quantiles, *zone_limits[zone])

    lstm = model.transform(lstm_estimates[zone].values, actuals[zone].values)
    lstm_observations.loc[:, idx[zone, "Normal"]] = lstm[0].squeeze()
    lstm_observations.loc[:, idx[zone, "CDF"]] = lstm[1].squeeze()

    taqr = model.transform(taqr_estimates[zone].values, actuals[zone].values)
    taqr_observations.loc[:, idx[zone, "Normal"]] = taqr[0].squeeze()
    taqr_observations.loc[:, idx[zone, "CDF"]] = taqr[1].squeeze()

lstm_observations.to_csv(save_path / "lstm_observations.csv")
lstm_observations.to_pickle(save_path / "lstm_observations.pkl")

taqr_observations.to_csv(save_path / "taqr_observations.csv")
taqr_observations.to_pickle(save_path / "taqr_observations.pkl")
