
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import properscoring as ps

import pathlib

import Code.quantiles as qm
import Code.misc as misc

from pandas import IndexSlice as idx

#stolen from nabqr
def variogram_score(x, y, p=0.5, window = 24, offset = 24):
    """Calculate the Variogram score for all observations for the time horizon t1 to t2.
    Modified from the R code in Energy and AI paper: 
    "An introduction to multivariate probabilistic forecast evaluation" by Mathias B.B. et al.
    Here we use t1 -> t2 as our forecast horizon.
    
    Parameters
    ----------
    x : numpy.ndarray
        Ensemble forecast (m x k)
    y : numpy.ndarray
        Actual observations (k,)
    p : float, optional
        Power parameter, by default 0.5
    t1 : int, optional
        Start hour (inclusive), by default 12
    t2 : int, optional
        End hour (exclusive), by default 36

    Returns
    -------
    tuple
        (score, score_list) Overall score and list of individual scores
    """
    m,n = x.shape
    
    score = 0
    variogram = np.zeros((window,window))
    n_windows = (n - offset) // window 
    
    for start in range(offset, n, window):
        
        if start + window > n: break
        
        for i in range(0,window-1):
            for j in range(i+1, window):
                diff = j-i
                Ediff =  np.mean(np.abs(x[:, start + i] - x[:, start + j])) ** p
                Adiff = np.abs(y[start + i] - y[start + j]) ** p
                               
                s = 1/diff * (Adiff - Ediff) ** 2
                variogram[i,j] = s
                variogram[j,i] = s
                score += s


    return score / n_windows, variogram / n_windows


def get_forecast(modelres, resids, N_step = 24, N_sim = 0, alpha = 0.1):
    # makes an n-step forecast using the fitter model
    #    model should ideally be refit every forecast, but SARMA model are relatively simple, so should be fine 
    # model: (for now) stastmodels SARIMAX model fit result
    # resids: (n,) array of residuals in normal space
    # N: number of data points to forecast
    # Return_sim: number of forecast simulations to make
    
    res = np.zeros((len(resids), 3))
    sim_res = np.zeros((len(resids), N_sim))
    
    #simulate first chunk if needed
    
    if N_sim > 0: 
        sim_res[:N_step, :] = modelres.simulate(nsimulations = N_step, repetitions = N_sim).squeeze()
    
    for k in range(N_step, len(resids) - N_step, N_step):
        # seems to take a bit longer to fit, all data might not be need
        
        min_k = max(k - 10 * N_step,0)
        res2 = modelres.apply(resids[min_k:k])
        
        tmp = res2.get_forecast(N_step)
        res[k:k + N_step,:] = tmp.summary_frame(alpha = alpha).values[:,[0,2,3]] # discard mean_se
        
        if N_sim > 0:
            sim_res[k:k + N_step, :] = res2.simulate(nsimulations = N_step, repetitions = N_sim, anchor = "end").squeeze()
    
    #get last chunk if needed
    last = (len(resids) // N_step) * N_step
    k_last = len(resids) - last
    
    if k_last < N_step:
        res2 = modelres.apply(resids[:last])
        
        tmp = res2.get_forecast(k_last)
        res[last:,:] = tmp.summary_frame(alpha = alpha).values[:,[0,2,3]] # discard mean_se
        if N_sim > 0:
            sim_res[last:, :] = res2.simulate(nsimulations = k_last, repetitions = N_sim, anchor = "end").squeeze()
            
    return res, sim_res

def calc_scores(actuals, predicted, simulations, **kwargs):
    #maybe return dict?
    
    MAE = np.mean(np.abs(predicted - actuals))
    MSE = np.mean((predicted - actuals)**2)
    VARS = variogram_score(sim.values.T, actuals.squeeze(), **kwargs)[0]
    CRPS = np.mean(ps.crps_ensemble(actuals.squeeze(), sim.values))
    
    return MAE, MSE, VARS, CRPS

def correction(actuals, est_quantiles, quant_est, model_type = "SARMA", N_step = 24, N_sim = 100, eval_resids = None):
    
    resids, _ = quant_est.transform(est_quantiles, actuals)
    resids = resids.squeeze()
    
    if model_type == "SARMA":
        model = sm.tsa.SARIMAX(resids, order = (1,0,1), seasonal_order=(1,0,1,24))
    elif model_type == "ARMA":
        model = sm.tsa.SARIMAX(resids, order = (1,0,1))
    model_res = model.fit()
    
    normal_space = get_forecast(model_res, resids, N_step = N_step, N_sim = N_sim)
    
    orignal_forecast, cdf_forecast = quant_est.back_transform(est_quantiles, normal_space[0])
    original_sim, cdf_sim = quant_est.back_transform(est_quantiles, normal_space[1])
    
    if eval_resids is not None:
        save_path = PATH / "Figures" / "Correlation Structure" / "Residuals"
        tmp = model_res.summary().tables[1].as_csv()
        with open(save_path / (eval_resids + "_" + model_type + "_coefs.csv"), 'w') as f:
            f.write(tmp)
                
        pr = stats.norm().cdf(model_res.resid / np.sqrt(model_res.mse))
        misc.evaluate_pseudoresids(pr[1:],
                                   save_path = save_path,
                                   name = eval_resids + "_after",
                                   close_figs = True
                                   )
    
    return normal_space, (cdf_forecast, cdf_sim), (orignal_forecast, original_sim)

#%% load data
PATH = pathlib.Path()

data = pd.read_pickle( PATH / "Data" / ("NABQR_results_full.pkl" ))
zones = data.columns.get_level_values(level = 0).unique()
quantiles = data[:]["Quantiles"].columns.unique().values
# there is some quantile crossings
# i will try to fix by just sorting
# might work?
for zone in zones:
    data[zone, "Quantiles"] = np.sort(data[zone, "Quantiles"].values)

models = ["SARMA", "ARMA"]
N_step = 24
N_sim = 300
    
df_forecast = pd.DataFrame(index = data.index,
                           columns = pd.MultiIndex.from_product([zones, ["nabqr"] + models, ("normal", "cdf", "original"), ("estimate", "lower prediction", "upper prediction")]),
                           dtype = np.float64
)

df_sim = pd.DataFrame(index = data.index,
                           columns = pd.MultiIndex.from_product([zones, ["nabqr"] + models, ("normal", "cdf", "original"), np.arange(N_sim)]),
                           dtype = np.float64
)
df_resids = pd.DataFrame(index = data.index,
                         columns = pd.MultiIndex.from_product([zones, ("normal", "cdf", "original")]),
                         dtype = np.float64
)

#%% naqbqr

for par, val in zip(("estimate", "lower prediction", "upper prediction"), (0.5,0.05,0.95)):
    df_forecast.loc[:, idx[:,"nabqr","normal", par]] = stats.norm().ppf(val)
    
df_sim.loc[:, idx[:, "nabqr", "normal", :]] = stats.norm().rvs((len(data), len(zones) * N_sim))

#convert to original space
for zone in zones:
    actuals = data[zone,"Observed"].values.squeeze()
    est_quantiles = data[zone, "Quantiles"].values.squeeze()
    quant_est = qm.piecewise_linear_model(quantiles, actuals.min() - 1, actuals.max()*1.1)
    
    tmp = quant_est.back_transform(est_quantiles, df_forecast.loc[:, idx[zone,"nabqr","normal", :]])
    df_forecast.loc[:, idx[zone,"nabqr","original", :]], df_forecast.loc[:, idx[zone,"nabqr","cdf", :]] = tmp
    
    tmp = quant_est.back_transform(est_quantiles, df_sim.loc[:, idx[zone,"nabqr","normal", :]])
    df_sim.loc[:, idx[zone,"nabqr","original", :]], df_sim.loc[:, idx[zone,"nabqr","cdf", :]] = tmp
    
    #also fill resid while we are at it
    tmp = quant_est.transform(est_quantiles, actuals)
    
    df_resids.loc[:, idx[zone, "original"]] = actuals
    df_resids.loc[:, idx[zone, "normal"]] = tmp[0].squeeze()
    df_resids.loc[:, idx[zone, "cdf"]] = tmp[1].squeeze()
    
    #evaluate resids
    misc.evaluate_pseudoresids(df_resids.loc[:, idx[zone, "cdf"]].values.squeeze(),
                               save_path = (PATH / "Figures" / "Correlation Structure" / "Residuals" ),
                               name = zone + "_before",
                               close_figs = True
                               )
    
#%%

for zone in zones:
    print(zone)
    for model in models:
        print(model)
        actuals = data[zone,"Observed"].values.squeeze()
        est_quantiles = data[zone, "Quantiles"].values.squeeze()
        quant_est = qm.piecewise_linear_model(quantiles, actuals.min() - 1, actuals.max()*1.1)
        
        normal, cdf, original = correction(actuals, est_quantiles, quant_est, model_type = model, N_step = N_step, N_sim = N_sim, eval_resids= zone + "_" + str(model) + "test")
        
        df_forecast.loc[:, idx[zone, model, "normal", :]] = normal[0]
        df_forecast.loc[:, idx[zone, model, "cdf", :]] = cdf[0]
        df_forecast.loc[:, idx[zone, model, "original", :]] = original[0]
        
        df_sim.loc[:, idx[zone, model, "normal", :]] = normal[1]
        df_sim.loc[:, idx[zone, model, "cdf", :]] = cdf[1]
        df_sim.loc[:, idx[zone, model, "original", :]] = original[1]
        

#%% scores
df_scores = pd.DataFrame(index = ["nabqr"] + list(models),
                         columns = pd.MultiIndex.from_product([zones,["MAE", "MSE", "VARS", "CRPS"]]),
                         dtype = float)

for zone in zones:
    for p in df_scores.index:
        
        actuals = data[zone,"Observed"].values.squeeze()
        predicted = df_forecast.loc[:,idx[zone, p, "original", "estimate"]].values
        sim = df_sim.loc[:, idx[zone, p, "original", :]]
        
        scores = calc_scores(actuals, predicted, sim.values)
        
        df_scores.loc[p, idx[zone, :]] = scores


#%% forecasts in orignal space


plots = [
    {
     "zone": "DK1-onshore",
     "space": "original",
     "xlims": (np.datetime64("2024-08-05"), np.datetime64("2024-08-14")),
     "ylims": (-100, 3100)
    },
    {
     "zone": "DK1-onshore",
     "space": "cdf",
     "xlims": (np.datetime64("2024-08-05"), np.datetime64("2024-08-14")),
     "ylims": (0,1)
    },
    {
     "zone": "DK1-onshore",
     "space": "normal",
     "xlims": (np.datetime64("2024-08-05"), np.datetime64("2024-08-14")),
     "ylims": (-3,3)
    },
    {
     "zone": "DK2-offshore",
     "space": "original",
     "xlims": (np.datetime64("2024-08-08"), np.datetime64("2024-08-18")),
     "ylims": (-100, 1100)
    },
    {
     "zone": "DK2-offshore",
     "space": "cdf",
     "xlims": (np.datetime64("2024-08-08"), np.datetime64("2024-08-18")),
     "ylims": (0,1)
    },
    {
     "zone": "DK2-offshore",
     "space": "normal",
     "xlims": (np.datetime64("2024-08-08"), np.datetime64("2024-08-18")),
     "ylims": (-3,3)
    },
]

def line_plots(zone, space, xlims, ylims, N_step = 24, save = True, Name = None, plot_smaller = False):
    # function for making this specific plot
    # not general, just saves a lot of writing
    x = df_forecast.index
    
    fig, ax = plt.subplots(figsize = (14,8), layout = "tight")
    sns.scatterplot(x = x, y = df_resids[zone, space], color = 'black', marker = 'x', ax = ax)
    sns.lineplot(x = x, y = df_forecast[zone, "SARMA", space, "estimate"], color = 'crimson', ax = ax)
    sns.lineplot(x = x, y = df_forecast[zone, "ARMA", space, "estimate"], color = 'olivedrab', ax = ax)
    sns.lineplot(x = x, y = df_forecast[zone, "nabqr", space, "estimate"], color = 'navy', ax = ax)
    ax.fill_between(
        x,
        df_forecast[zone, "SARMA", space, "lower prediction"],
        df_forecast[zone, "SARMA", space, "upper prediction"],
        color = 'crimson',
        alpha = 0.3
    )
    ax.fill_between(
        x,
        df_forecast[zone, "nabqr", space, "lower prediction"],
        df_forecast[zone, "nabqr", space, "upper prediction"],
        color = 'navy',
        alpha = 0.3
    )
    ax.vlines(
        data.index[N_step-1::N_step],
        np.nanmin(df_forecast[zone].values),
        np.nanmax(df_forecast[zone].values),
        color = 'black',
        linestyle = '--'
    )
    ax.set_title(zone)
    ax.set_ylabel("Production")
    ax.set_xlabel("Date")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    if save:
        if Name is None:  Name = ("-".join([zone, space, "forecast"]))
        fig.savefig( (PATH / "Figures" / "Correlation Structure" / Name).with_suffix(".png") )
        
    return fig, ax

for kwargs in plots:
    line_plots(plot_smaller = True, **kwargs)
    


#%% actual by predicted

def abp_plot(zone, space, xlim, ylim, save = True, Name = None):
    # function for making this specific plot
    # not general, just saves a lot of writing

    fig, ax = plt.subplots(figsize = (14,8), layout = "tight")
    
    sns.scatterplot(
        x = df_forecast[zone, "SARMA", space, "estimate"],
        y = df_resids[zone, space],
        color = 'crimson',
        ax = ax
    )
    sns.scatterplot(
        x = df_forecast[zone, "ARMA", space, "estimate"],
        y = df_resids[zone, space],
        color = 'olivedrab',
        ax = ax
    )
    sns.scatterplot(
        x = df_forecast[zone, "nabqr", space, "estimate"],
        y = df_resids[zone, space],
        color = 'navy',
        ax = ax
    )
    ax.axline((0,0), slope = 1, color = 'black')
    ax.set_title(zone)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(["SARMA", "ARMA", "NABQR"])

    if save:
        if Name is None:  Name = ("-".join([zone, space, "actual_predicted"]))
        fig.savefig( (PATH / "Figures" / "Correlation Structure" / Name).with_suffix(".png") )
        
    return fig, ax


xlims = [
    (-10,1400),
    (-40,3700),
    (-10,1100),
    (-55,550)
]

ylims = [
    (-10,1200),
    (-10,3400),
    (-10,1000),
    (-10,580)
]
for zone, xlim, ylim in zip(zones, xlims, ylims):
    abp_plot(zone, "original", xlim, ylim)
    abp_plot(zone, "cdf", (0,1), (0,1))
    abp_plot(zone, "normal", (-3,3), (-3,3))
    

#%% scores 
tmp = df_scores.T.reset_index(names = ["zone", "metric"]).melt(value_vars = ["nabqr", "SARMA", "ARMA"],
                                                               var_name = "model",
                                                               value_name ="score",
                                                               id_vars = ["zone","metric"])
fig = sns.catplot(tmp,
                x = "model", y = "score", hue = "model",
                row = "metric", col = "zone", kind = "bar",
                palette = {"nabqr": "navy", "SARMA":"crimson", "ARMA":"olivedrab"},
                height = 3, aspect = 1.2,
                sharey = False, sharex=False,
                margin_titles = True,
                dodge = False)
fig.tight_layout()

fig.savefig( (PATH / "Figures" / "Correlation Structure" / "scores.png" ))


#%% variogram

fig, axes = plt.subplots(len(zones), len(df_scores.index),
                         figsize = (12,14), layout = "constrained",
                         sharex = True, sharey = True)

c_range = []
for i, zone in enumerate(zones):
    for j, p in enumerate(df_scores.index):
        
        actuals = data[zone,"Observed"].values.squeeze()
        sim = df_sim.loc[:, idx[zone, p, "original", :]]
        
        variogram = variogram_score(sim.values.T, actuals)[1]
        
        if p == "nabqr": c_range.append((0,variogram.max()))
        
        cbar = False
        if j == len(df_scores.index)-1: cbar = True
        sns.heatmap(variogram,
                    ax = axes[i,j],
                    vmin = c_range[i][0], vmax = c_range[i][1],
                    xticklabels = 1, yticklabels = 1,
                    cmap = "viridis",
                    cbar = cbar)
        
        if i == 0: axes[i,j].set_title(p)
        if j == 0: axes[i,j].set_ylabel(zone)

fig.savefig( (PATH / "Figures" / "Correlation Structure" / "variograms.png" ))

