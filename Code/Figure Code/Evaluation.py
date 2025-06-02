import pathlib
import tomllib

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import properscoring as ps
import seaborn as sns

import nabqra


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Evaluation"
save_path = PATH / "Results" / "Evaluation"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)

rng = np.random.default_rng(42)

plt.close("all")


# %% distance metrics

dist1 = stats.norm()
dist2 = stats.t(4)

X = np.linspace(-6, 6, num=10000)

# distributions for comparison
fig, ax = plt.subplots()

lines = ax.plot(X, np.array([dist1.pdf(X), dist2.pdf(X)]).T)
ax.fill_between(X, dist1.pdf(X), color=lines[0].get_color(), alpha=0.5, label=r"$x \sim F = N(0,1^2)$")
ax.fill_between(X, dist2.pdf(X), color=lines[1].get_color(), alpha=0.5, label=r"$x \sim G = t(4)$")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("p(x)")

fig.savefig(save_path / "Figures" / "Example Dists")
plt.close(fig)

# %% wasserstein + lp distance
fig, ax = plt.subplots()

ax.plot(X, np.array([dist1.ppf(X), dist2.ppf(X)]).T)
ax.fill_between(X, dist1.ppf(X), dist2.ppf(X), color="grey", alpha=0.3)
ax.set_ylim([-3, 3])
ax.legend(["$F = N(0,1^2)$", "$G = t(4)$", "Wasserstein Area"])
ax.set_ylabel("Quantile(x)")
ax.set_xlabel("x")

fig.savefig(save_path / "Figures" / "Wasserstein Area")
plt.close(fig)

fig, ax = plt.subplots()

wasser_table = pd.DataFrame(index=["$W_p^p$", "$W_p$"])
Q = np.linspace(1e-8, 1 - 1e-8, num=1000)

for p in [0.1, 0.5, 1, 2]:
    d = np.abs(dist1.ppf(Q) - dist2.ppf(Q)) ** p

    score = nabqra.scoring.continous_wasserstein(dist1.ppf, dist2.ppf, order=p, lims=[1e-8, 1 - 1e-8], limit=200)[0]
    wasser_table[f"{p:.1f}"] = score**p, score

    nabqra.plotting.band_plot(Q, d, d, band_label=f"$ p = {p} $", ax=ax)


ax.set_xlabel("q")
ax.set_ylabel(r"$W_p^p$")
ax.set_ylim([0, 2])
ax.set_xlim([0, 1])
ax.legend()

fig.savefig(save_path / "Figures" / "Wasserstein Distance")
plt.close(fig)

wasser_table.style.to_latex(
    save_path / "Tables" / "Wasserstein.tex",
    position="h",
    position_float="centering",
    label="evaluation:table:wasserstein",
    caption=(
        "Wasserstein distance for different orders. Largers orders Emphasize larger deviations",
        "Wasserstein distance for different orders",
    ),
    hrules=True,
)

# %% kl Divergence

X = np.linspace(-3, 3, num=10000)
X2 = np.linspace(-12, 12, num=10000)

fig, ax = plt.subplots()

a1 = ax.fill_between(
    X, -np.log(dist1.pdf(X)) + np.log(dist2.pdf(X)), color=lines[0].get_color(), label=r"$I_F(x) - I_G(x)$"
)
a2 = ax.fill_between(
    X, -np.log(dist2.pdf(X)) + np.log(dist1.pdf(X)), color=lines[1].get_color(), label=r"$I_G(x) - I_F(x)$"
)

ax.set_xlabel("x")
ax.set_ylabel("Excess Information")
ax.legend()

fig.savefig(save_path / "Figures" / "Excess Information")
plt.close(fig)


fig, ax = plt.subplots()
kl1 = dist1.pdf(X2) * np.log(dist1.pdf(X2) / dist2.pdf(X2))
kl2 = dist2.pdf(X2) * np.log(dist2.pdf(X2) / dist1.pdf(X2))
ax.fill_between(X2, kl1, label=r"$D_{{kl}}(G\ |F)$")
ax.fill_between(X2, kl2, label=r"$D_{{kl}}(F\ |G)$")
ax.plot(X2, kl1 + kl2, color="black", label=r"Amplitude Difference")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("KL-Divergence")

fig.savefig(save_path / "Figures" / "KL-Divergence")
plt.close(fig)


scores = np.array(
    [
        nabqra.scoring.continous_kl_divergence(dist1.pdf, dist2.pdf, (-30, 30), limit=200)[0],
        nabqra.scoring.continous_kl_divergence(dist2.pdf, dist1.pdf, (-30, 30), limit=200)[0],
    ]
)
kl_table = pd.DataFrame(np.atleast_2d(scores), index=["Divergence"], columns=["$D_{{kl}}(G\ |F)$", "$D_{{kl}}(F\ |G)$"])
kl_table.style.to_latex(
    save_path / "Tables" / "Kullback-Leibler.tex",
    position="h",
    position_float="centering",
    label=r"evaluation:table:kullback-leibler",
    caption=(
        "Kullback-Leibler divergence between the distributions"
        "$F = N(0,1^2)$ and $G = t(4)$."
        "Notice the divergence is not symmetrical.",
        "Kullback-Leibler divergence",
    ),
    hrules=True,
)

# %% MAE

X = np.linspace(-3, 3, num=1000)
X2 = np.array([1, 10, 20])

Ys = np.add.outer(np.abs(X), X2) / 2

fig, ax = plt.subplots()

ax.plot(X, Ys)

ax.legend([f"$x_2 = {x2}$" for x2 in X2])
ax.set_xlabel("$x_1$")
ax.set_ylabel("MAE")

fig.savefig(save_path / "Figures" / "MAE")
plt.close(fig)

# %% RMSE

X = np.linspace(-3, 3, num=1000)
X2 = np.array([1, 10, 100])

Ys = np.sqrt(np.add.outer(X**2, X2**2) / 2)
Y_max = Ys.max(axis=0)

labels = [f"$x_2 = {x2}$" for x2 in X2]

fig, ax = nabqra.plotting.multi_y_plot(X, Ys.T, labels=labels, ylims=list(zip(Y_max - 1.8, Y_max)))

ax.get_legend().set(loc="lower left")
ax.set_xlabel("$x_1$")
ax.set_title("RMSE")

fig.savefig(save_path / "Figures" / "RMSE")
plt.close(fig)

# %% QS


def quantile_loss(q, y, f):
    e = y - f
    return np.maximum(q * e, (q - 1) * e)


quantiles = [0.001, 0.5, 0.95]

X = np.linspace(-4, 4, num=1000)
Y = np.zeros((len(quantiles), len(X)))

fig, ax = plt.subplots()

for q in quantiles:
    loss = quantile_loss(q, X, stats.norm.ppf(q))
    l = ax.plot(X, loss, label=f"q = {q}")

ax.set_xlabel("$x$")
ax.set_ylabel("$Quantile Loss$")
ax.legend()

fig.savefig(save_path / "Figures" / "QS")
plt.close(fig)


# %% CRPS

dist = stats.norm()
example_point = -0.5
X = np.linspace(-3, 3, 1000)
sigmas = [0.1, 1, 10]
Ys = np.array([ps.crps_gaussian(X, 0, sigma) for sigma in sigmas]).T


fig, ax = plt.subplots()

ax.plot(X, dist.cdf(X), color="black", label=r"True distribution")
ax.plot(X, np.heaviside(X - example_point, 1), label="Observed Distribution")
ax.vlines(example_point, 0, 1, color="black", linestyle="--", label=r"Observed x")
ax.fill_between(X, dist.cdf(X), np.heaviside(X - example_point, 1), alpha=0.3)

ax.set_xlabel("$x$")
ax.set_ylabel("$F(x)$")
ax.legend()

fig.savefig(save_path / "Figures" / "CRPS observation")
plt.close(fig)


fig, ax = plt.subplots()
ax.plot(X, Ys)
ax.legend([f"$N(0,{sigma}^2)$" for sigma in sigmas])
ax.set_ylabel("$CRPS$")
ax.set_xlabel("$x$")

fig.savefig(save_path / "Figures" / "CRPS")
plt.close(fig)

# %% Vars

theoretical = pd.read_pickle(load_path / "Theoretical.pkl")
samples = pd.read_pickle(load_path / "Samples.pkl")


fig, ax = nabqra.plotting.band_plot(
    theoretical.index,
    theoretical["Expected"],
    theoretical["Lower"],
    theoretical["Upper"],
    label=r"$E(x_t)$",
    band_label="90% interval",
)
ax.plot(samples.index, samples.values[:, :12], color=ax.get_lines()[0].get_color(), alpha=0.4)
ax.set_xlabel("$t$")
ax.set_ylabel("$x_t$")
plt.legend()
fig.savefig(save_path / "Figures" / "Process example")
plt.close(fig)

# %% vars

diffs = samples - samples.iloc[0, :]
abs_diffs = np.abs(diffs)
X = samples.index

fig, ax = nabqra.plotting.band_plot(X, theoretical["Expected"], theoretical["Upper"], label="$E(x_t - x_0)$")

for p in [0.5, 1, 2]:
    ax.plot(X, np.mean(abs_diffs**p, axis=1), label=f"$E(|x_t - x_0|^{{{p}}})$")

ax.set_xlabel("$t$")
ax.legend()
fig.savefig(save_path / "Figures" / "VarS diff")
plt.close(fig)


# %% vars

tmp_df = pd.DataFrame({f"p={p}": abs_diffs.iloc[-1, :] ** p for p in [0.5, 1, 2]}).melt(
    var_name="Order", value_name="x"
)

fig, ax = plt.subplots()
sns.kdeplot(
    tmp_df,
    x="x",
    hue="Order",
    cut=0,
    fill=True,
    multiple="layer",
    ax=ax,
)
ax.set_xlabel("$|x_t - x_0|^{{{p}}}$")
ax.set_ylabel("p(x)")
ax.set_xlim([0, 10])
fig.savefig(save_path / "Figures" / "VarS Distribution")
plt.close(fig)


# %% Expected score as a function of variance

sigmas = np.logspace(-2, 2, 100)
N = 10000
K = 10
z = stats.norm(scale=sigmas).rvs((K, N, len(sigmas)), random_state=rng)

mae = np.mean(np.abs(z), axis=0)
rmse = np.sqrt(np.mean(z**2, axis=0))
crps = np.mean([ps.crps_gaussian(z[:, :, i], 0, s) for i, s in enumerate(sigmas)], axis=1).T

VarS_mean = np.mean(np.abs(z) ** 0.5, axis=(0, 1))
VarS = np.mean(np.abs(np.abs(z) ** 0.5 - VarS_mean) ** (1 / 0.5), axis=0)

fig, ax = plt.subplots()

ax.plot(sigmas, np.mean(mae, axis=0))
ax.plot(sigmas, np.mean(rmse, axis=0))
ax.plot(sigmas, np.mean(crps, axis=0))
ax.plot(sigmas, np.mean(VarS, axis=0))

plt.legend(["MAE", "RMSE", "CRPS", "Vars"])
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Expected Score")
ax.set_xlabel(r"$\sigma$")

fig.savefig(save_path / "Figures" / "Expected Score")
plt.close(fig)
