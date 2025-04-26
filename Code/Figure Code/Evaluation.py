import pathlib
import tomllib

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import properscoring as ps
import seaborn as sns

import nabqra.scoring as scoring


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data"
save_path = PATH / "Results" / "Evaluation" 

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    
    
colors =  plt.rcParams['axes.prop_cycle'].by_key()["color"]
# small function for calculating wasserstein dist
# only for plotting, dont use elsewhere
def wasser(x, y, p):
    return np.abs(x - y)**p

def multi_y_plot(X, Ys, ax = None,
                 labels = None, ylims = None,
                 offset = 0.05, hide_original = True,
                 color_cycler = None,
                 label_pos = "Legend"):
    
    if ax is None:
        fig, ax = plt.subplots()
        
    else:
        fig = ax.get_figure()
        
    if labels is None:
        labels = [None for _ in range(len(Ys))]
    if ylims is None:
        ylims = [None for _ in range(len(Ys))]
    if color_cycler is None:
        color_cycler = plt.rcParams['axes.prop_cycle']()
        
    if hide_original:
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        
    offsets = [-offset*x for x in range(len(Ys))]
    
    lines = []
    for y, label, ylim, offset, color in zip(Ys, labels, ylims, offsets, color_cycler):
        
        color = color["color"]
        
        twin = ax.twinx()
        
        line = twin.plot(X, y, color = color)
        lines.append(line[0])
        
        twin.spines["left"].set_visible(True)
        twin.spines["left"].set_position(("axes", offset))
        twin.spines["left"].set_color(color)
        
        twin.yaxis.set_label_position('left')
        twin.tick_params(axis = "y", which = "both", colors = color)
        
        twin.yaxis.set_ticks_position('left')
        twin.set_ylim(ylim)
        
        if label_pos == "Axis": 
            twin.set_ylabel(label, labelpad = 0, color = color)
        else:
            twin.set_ylabel("", labelpad = 0, color = color)
    
    if label_pos == "Legend": 
        ax.legend(lines, labels)
    
    return fig, ax


def band_plot(x, y_mid, y1, y2 = 0, ax = None,
              color = None, alpha = 0.3, label = None, band_label = None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        
    p = ax.plot(x, y_mid, label = label)
    color = p[0].get_color()
    
    ax.fill_between(x, y1, y2, color = color, alpha = alpha, label = band_label)

    return fig, ax

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
#plt.close(fig)


fig, ax = plt.subplots()
kl1 = dist1.pdf(X2) * np.log(dist1.pdf(X2) / dist2.pdf(X2))
kl2 = dist2.pdf(X2) * np.log(dist2.pdf(X2) / dist1.pdf(X2))
ax.fill_between(X2, kl1, label = "$D_{{kl}}(G\ |F)$", alpha = 0.8)
ax.fill_between(X2, kl2, label = "$D_{{kl}}(F\ |G)$", alpha = 0.8)
ax.plot(X2, kl1 + kl2, color = "black", label = "$|D_{{kl}}(G\ |F)| - |D_{{kl}}(F\ |G)|$")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("KL-Divergence")

fig.savefig(save_path / "Figures" / "KL-Divergence")
#plt.close(fig)


scores = np.array([
    scoring.continous_kl_divergence(dist1.pdf, dist2.pdf, (-30,30), limit = 200)[0],
    scoring.continous_kl_divergence(dist2.pdf, dist1.pdf, (-30,30), limit = 200)[0]
])
kl_table = pd.DataFrame(np.atleast_2d(scores), index = ["Divergence"], columns = ["$D_{{kl}}(G\ |F)$", "$D_{{kl}}(F\ |G)$"])
kl_table.style.to_latex(save_path / "Tables" / "Kullback-leibler.tex",
                            position = "h",
                            label = "evaluation:table:kullback-leibler",
                            caption = ('Kullback-Leibler divergence between the distributions'
                                       '$F = N(0,1^2)$ and $G = t(4)$.'
                                       'Notice the divergence is not symmetrical.',
                                       "Kullback-Leibler divergence"),
                            hrules = True)

#%% MAE

X = np.linspace(-3,3, num = 1000)

X2 = [1,5,10,50]
Ys = [np.abs(X) + x2 for x2 in [1,5,10,50]]
labels = [f'$x_2 = {x2}$' for x2 in X2]
ylims = [(0, X[-1] + x2) for x2 in X2]

fig,ax = multi_y_plot(X, Ys, labels = labels, ylims = ylims, offset = 0.03)

ax.get_legend().set(loc = "lower right")
ax.set_title("MAE")
ax.set_xlabel("$x1$")

#%% RMSE

X = np.linspace(-3,3, num = 1000)
X2 = np.logspace(0,2,3)

Ys = [np.sqrt((X**2 + x2**2) / 2) for x2 in X2]
labels = [f'$x_2 = {x2}$' for x2 in X2]

dy = Ys[0][-1]
ylims = [(y[-1] - dy, y[-1] + dy/2) for y in Ys]

fig,ax = multi_y_plot(X, Ys, labels = labels, ylims = ylims, offset = 0.05)

ax.get_legend().set(loc = "lower right")
ax.set_title("RMSE")
ax.set_xlabel("$x1$")


#%% CRPS

dist = stats.norm()
example_point = -0.5
X = np.linspace(-3,3, 1000)

fig, ax = plt.subplots()

ax.plot(X, dist.cdf(X), color = "black", label = "True distribution")
ax.vlines(example_point, 0, 1, color = "black", linestyle = "--", label = "Observed x")
ax.plot(X, np.heaviside(X - example_point, 1), label = '"Observed" Distribution')
ax.fill_between(X, dist.cdf(X), np.heaviside(X - example_point, 1), alpha = 0.3, label = "CRPS")
ax.set_xlabel("$x$")
ax.set_ylabel("$F(x)$")
ax.set_title("CRPS calculation")
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
ax.set_title("$CRPS$")
ax.set_xlabel("$x$")



#%% Vars
# Thomas studienummber laver en pæn graf
rng = np.random.default_rng(212516)

N = 10000
T = 15
N_steps = 100

ar1 = 0.75**(1/N_steps)
sigma = 1 / N_steps
initial = -1
alpha = 0.1


model = sm.tsa.ARIMA([0,0], order = (1,0,0), trend = 'n')

samples = model.simulate([ ar1, sigma], T*N_steps, repetitions=N, initial_state=initial, random_state = rng).squeeze()
samples = samples - initial

X = np.linspace(0, T, num = T*N_steps)

diffs = samples - samples[0,:]
abs_diffs = np.abs(diffs)

expected = np.zeros_like(X)
expected[0] = initial
var = np.zeros_like(X)
for i in range(1,len(X)):
    expected[i] = ar1*expected[i-1]
    var[i] = ar1**2 * var[i-1] + sigma
expected = expected - initial
process_interval = stats.norm().ppf(1-alpha/2)*np.sqrt(var)

# process plot
fig, ax = band_plot(X,
                    expected,
                    expected - process_interval,
                    expected + process_interval,
                    label = "$E(x_t)$", band_label=f'$ {alpha*100:.0f}\%$ interval'
)
ax.plot(X, samples[:, :12], color = ax.get_lines()[0].get_color(), alpha = 0.4)

fig, ax = plt.subplots()
ax.plot(X, expected, color = "black", label = '$E(x_t- x_0)$')
p = ax.plot(X, np.mean(abs_diffs, axis = 1), label = '$E(|x_t - x_0|)$')
band_plot(X,
             *np.quantile(abs_diffs, [0.5,0.90], axis = 1),0,
             color = p[0].get_color(),
             label = '$Q_{0.5}(|x_t - x_0|)$',
             band_label = f'${alpha*100:.0f}\%$ interval',
             ax = ax)
ax.legend()

diffs_df = pd.DataFrame(diffs[0,:], columns = ["observation"])
abs_diffs_df = pd.DataFrame(abs_diffs[-1,:], columns = ["observation"])
diffs_df["dist"] = "diff"
abs_diffs_df["dist"] = "abs_diff"

fig, ax = plt.subplots()
sns.histplot(pd.concat((diffs_df,abs_diffs_df), ignore_index = True),
             x = "observation", hue = "dist",
             kde = True, stat = "density", multiple = "layer",
             ax = ax)

#%% Expected score as a function of variance

sigmas = np.sqrt(np.logspace(-4,4,100))
N = 10000
K = 10
z = stats.norm(scale = sigmas).rvs(( K, N,  len(sigmas)))

mae = np.mean(np.abs(z), axis = 0)
rmse = np.sqrt(np.mean(z**2, axis = 0))
Vars = np.mean(np.abs(np.abs(z) - np.mean(np.abs(z)**0.5, axis = (0,1)))**0.5, axis = 0)
crps = np.mean([ps.crps_gaussian(z[:,:,i], 0, s) for i, s in enumerate(sigmas)], axis = 1).T


def mmq_plot(x, y, ax, alpha = 0.05, sided = 2):
    ax.plot(x, np.mean(y, axis = 0), linewidth = 2)


fig, ax = plt.subplots()

mmq_plot(sigmas, mae, ax)
mmq_plot(sigmas, rmse, ax)
mmq_plot(sigmas, crps, ax)
mmq_plot(sigmas, Vars, ax)
plt.legend(["MAE", "RMSE", "CRPS", "Vars"])
ax.set_yscale("log")
ax.set_xscale("log")

