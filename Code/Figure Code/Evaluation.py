import pathlib
import tomllib

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import properscoring as ps

import nabqra.scoring as scoring

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data"
save_path = PATH / "Results" / "Evaluation" 

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    
    
# small function for calculating wasserstein dist
# only for plotting, dont use elsewhere
def wasser(x, y, p):
    return np.abs(x - y)**p

plt.close('all')
#%% distance metrics

dist1 = stats.norm()
dist2 = stats.t(4)

Q = np.linspace(1e-12,1-1e-12, num = 10000)
X = np.linspace(-6,6, num = 10000)


# distributions for comparison
fig, ax = plt.subplots()

lines = ax.plot(X, np.array([dist1.pdf(X), dist2.pdf(X)]).T)
ax.fill_between(X, dist1.pdf(X), color = lines[0].get_color(), alpha = 0.5, label = "$x \sim F = N(0,1^2)$")
ax.fill_between(X, dist2.pdf(X), color = lines[1].get_color(), alpha = 0.5, label = "$x \sim G = t(2)$")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("p(x)")

fig.savefig(save_path / "Figures" / "Dists")
plt.close(fig)

#%% wasserstein
fig, ax = plt.subplots()

ax.plot(X, np.array([dist1.ppf(X), dist2.ppf(X)]).T)
ax.fill_between(X, dist1.ppf(X), dist2.ppf(X), color = 'grey', alpha = 0.3)
ax.set_ylim([-3,3])
ax.legend(["$F = N(0,1^2)$", "$G = t(2)$", "Wasserstein Area"])

fig.savefig(save_path / "Figures" / "Wasserstein Area")
plt.close(fig)

wasser_table = pd.DataFrame(index = ["Wasserstein Area", "Wasserstein_distance"])
fig, ax = plt.subplots()

for p in [0.1,0.5,1,2]:
    
    d = wasser(dist1.ppf(Q),dist2.ppf(Q), p)
    score = scoring.continous_wasserstein(dist1.ppf,dist2.ppf, [1e-8,1-1e-8], order = p, limit = 200)[0]
    
    wasser_table[f'{p:.1f}'] = score**p, score
    
    line = ax.plot(Q, d)[0]
    ax.fill_between(Q, d, color = line.get_color(), alpha = 0.2, label = f'$W_{{{p}}}$')
ax.set_ylim([0,2])
ax.set_xlim([0,1])
ax.legend()

fig.savefig(save_path / "Figures" / "Wassestein distance")
#plt.close(fig)

wasser_table.style.to_latex(save_path / "Tables" / "Wasserstein.tex",
                            position = "h",
                            label = "evaluation:table:wasserstein",
                            caption = ('Wasserstein distance for different orders. largers order Emphasise larger deviations',
                                       "Wasserstein distance for different orders"),
                            hrules = True)
#%% kl Divergence

X = np.linspace(-3,3, num = 10000)
X2 = np.linspace(-12,12, num = 10000)

fig, ax = plt.subplots()
ax2 = ax.twinx()
lines = ax2.plot(X, np.array([dist1.pdf(X), dist2.pdf(X)]).T, label = ["f(x)", "g(x)"])
a1 = ax.fill_between(X, np.log(dist1.pdf(X) / dist2.pdf(X)),
                     alpha = 0.8, color = lines[0].get_color(),
                     label = "Excess Information $G\ |F$")
a2 = ax.fill_between(X, np.log(dist2.pdf(X) / dist1.pdf(X)),
                     alpha = 0.8, color = lines[1].get_color(),
                     label = "Excess Information $F\ |G$")
ax.set_xlabel("x")
ax.set_ylabel("Excess Information")
ax.legend(lines + [a2, a1], [x.get_label() for x in lines + [a2, a1]] )

fig.savefig(save_path / "Figures" / "Excess information")
plt.close(fig)


fig, ax = plt.subplots()
kl1 = dist1.pdf(X2) * np.log(dist1.pdf(X2) / dist2.pdf(X2))
kl2 = dist2.pdf(X2) * np.log(dist2.pdf(X2) / dist1.pdf(X2))
ax.fill_between(X2, kl1, label = "$D_{{kl}}(G\ |F)$")
ax.fill_between(X2, kl2, label = "$D_{{kl}}(F\ |G)$")
ax.plot(X2, kl1 + kl2, color = "black", label = "$|D_{{kl}}(G\ |F)| - |D_{{kl}}(F\ |G)|$")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("Kullback-Leibler Divergence")

fig.savefig(save_path / "Figures" / "KL-Divergence")
plt.close(fig)


#%%
scores = np.array([
    scoring.continous_kl_divergence(dist1.pdf, dist2.pdf, (-30,30), limit = 200)[0],
    scoring.continous_kl_divergence(dist2.pdf, dist1.pdf, (-30,30), limit = 200)[0]
])
kl_table = pd.DataFrame(np.atleast_2d(scores), index = ["Divergence"], columns = ["$D_{{kl}}(G\ |F)$", "$D_{{kl}}(F\ |G)$"])
wasser_table.style.to_latex(save_path / "Tables" / "Kullback-leibler.tex",
                            position = "h",
                            label = "evaluation:table:kullback-leibler",
                            caption = ('Kullback-Leibler divergence between the distributions $F = N(0,1^2)$ and $G = t(4)$. Notice the divergence is not symmetrical',
                                       "Kullback-Leibler divergence"),
                            hrules = True)



#%% MAE + RMSE

#MAE
fig, ax = plt.subplots()
# remove original y axis
ax.spines["left"].set_visible(False)
ax.set_yticks([])

#this should be made into a function
lines = []
for x2, color, offset in zip([1, 5, 10, 50], plt.rcParams['axes.prop_cycle'].by_key()["color"], [0, -0.05,-0.1,-0.15] ):

    twin = ax.twinx()
    p = twin.plot(X, (np.abs(X) + x2) / 2, color = color, label = f'$x_2 = {x2}$')
    lines.append(p[0])

    twin.spines["left"].set_position(("axes", offset))
    twin.spines["left"].set_visible(True)
    twin.spines["left"].set_color(color)
    twin.yaxis.set_label_position('left')
    twin.yaxis.set_ticks_position('left')
    twin.tick_params(axis = "y", which = "both", colors = color)
    
    twin.set_ylabel("", labelpad = 0, color = color)
    twin.set_ylim([0,(X[-1] + x2) / 2])
ax.legend(lines, [x.get_label() for x in lines], loc = "lower right")
ax.set_title("MAE")
ax.set_xlabel("$x1$")
#RMSE
fig, ax = plt.subplots()
# remove original y axis
ax.spines["left"].set_visible(False)
ax.set_yticks([])

#this should be made into a function
lines = []
for x2, color, offset in zip([1, 5, 10, 50], plt.rcParams['axes.prop_cycle'].by_key()["color"], [0, -0.05,-0.1,-0.15] ):

    twin = ax.twinx()
    p = twin.plot(X, np.sqrt((X**2 + x2**2) / 2), color = color, label = f'$x_2 = {x2}$')
    lines.append(p[0])

    twin.spines["left"].set_position(("axes", offset))
    twin.spines["left"].set_visible(True)
    twin.spines["left"].set_color(color)
    twin.yaxis.set_label_position('left')
    twin.yaxis.set_ticks_position('left')
    twin.tick_params(axis = "y", which = "both", colors = color)
    
    twin.set_ylabel("", labelpad = 0, color = color)
    twin.set_ylim([0,np.sqrt((X[-1]**2 + x2**2) / 2)])
ax.legend(lines, [x.get_label() for x in lines], loc = "lower right")
ax.set_title("RMSE")
ax.set_xlabel("$x1$")


#%% CRPS

dist = stats.norm()
example_point = -0.5
X = np.linspace(-3,3, 1000)

fig, ax = plt.subplots()

ax.plot(X, dist.cdf(X), color = "black", label = "True distribution")
ax.plot(X, np.heaviside(X - example_point, 1), label = '"Observed" Distribution')
ax.fill_between(X, dist.cdf(X), np.heaviside(X - example_point, 1), alpha = 0.3, label = "CRPS")
ax.set_xlabel("$x$")
ax.set_ylabel("$F(x)$")
ax.legend()

fig, ax = plt.subplots()
# remove original y axis
ax.spines["left"].set_visible(False)
ax.set_yticks([])

#this should be made into a function
lines = []
for sigma, color, offset in zip([0.01,0.1,1,10], plt.rcParams['axes.prop_cycle'].by_key()["color"], [0, -0.05,-0.1,-0.15] ):

    twin = ax.twinx()
    p = twin.plot(X, ps.crps_gaussian(X, 0, sigma), color = color, label = f'$\sigma = {sigma}$')
    lines.append(p[0])

    twin.spines["left"].set_position(("axes", offset))
    twin.spines["left"].set_visible(True)
    twin.spines["left"].set_color(color)
    twin.yaxis.set_label_position('left')
    twin.yaxis.set_ticks_position('left')
    twin.tick_params(axis = "y", which = "both", colors = color)
    
    twin.set_ylabel("", labelpad = 0, color = color)
    twin.set_ylim([0, ps.crps_gaussian(X[-1], 0, sigma)])
ax.legend(lines, [x.get_label() for x in lines], loc = "lower right")
ax.set_title("CRPS")
ax.set_xlabel("$x1$")




#forecast
#MAE, RMSE CRPS, VARS, 

