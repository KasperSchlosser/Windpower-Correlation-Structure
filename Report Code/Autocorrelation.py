import pathlib
import os
os.chdir(pathlib.Path("..").resolve())


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


import Code.quantiles as qm
import Code.evaluation as evaluation

from pandas import IndexSlice as idx


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



def correction(actuals, est_quantiles, quant_est, model_type = "SARMA", train_proportion = 0.5, N_step = 24, N_sim = 100, eval_resids = None):
    
    resids, _ = quant_est.transform(est_quantiles, actuals)
    resids = resids.squeeze()
    
    n_train = int(len(resids)*train_proportion)
    
    if model_type == "SARMA":
        model = sm.tsa.SARIMAX(resids[:n_train], order = (1,0,1), seasonal_order=(1,0,1,24))
    elif model_type == "ARMA":
        model = sm.tsa.SARIMAX(resids[:n_train], order = (1,0,1))
    model_res = model.fit()
    test_res = model_res.apply(resids[n_train+1:], refit = False)
    
    normal_space = get_forecast(model_res, resids, N_step = N_step, N_sim = N_sim)
    
    orignal_forecast, cdf_forecast = quant_est.back_transform(est_quantiles, normal_space[0])
    original_sim, cdf_sim = quant_est.back_transform(est_quantiles, normal_space[1])
    
    if eval_resids is not None:
        save_path = PATH / "Figures" / "Correlation Structure" / "Residuals"
        tmp = model_res.summary().tables[1].as_csv() 
        with open(save_path / (eval_resids + "_" + model_type + "_coefs.csv"), 'w') as f:
            f.write(tmp)
        
        pr = stats.norm().cdf(test_res.resid / np.sqrt(test_res.mse))
        evaluation.evaluate_pseudoresids(pr,
                                   save_path = save_path,
                                   name = eval_resids + "_after",
                                   close_figs = True
                                   )
    
    return normal_space, (cdf_forecast, cdf_sim), (orignal_forecast, original_sim), n_train

#%% load data

PATH = pathlib.Path()

data = pd.read_pickle( PATH / "Data" / ("NABQR_results_full.pkl" ))
zones = data.columns.get_level_values(level = 0).unique()
quantiles = data[:]["Quantiles"].columns.unique().values

#zone min and max
#there seems to be a cap on production
#set slighty lower and higher to be safe, there is a bit of noise
zone_lims = {"DK1-offshore":(-1,1200),"DK1-onshore":(-1,3200),"DK2-offshore":(-1,1000),"DK2-onshore":(-1,550)}

# there is some quantile crossings
# i will try to fix by just sorting
# might work?
for zone in zones:
    data[zone, "Quantiles"] = np.sort(data[zone, "Quantiles"].values)

models = ["SARMA", "ARMA"]
N_step = 24
N_sim = 500
train_proportion = 0.5
n_train = int(len(data)*train_proportion)

index = pd.MultiIndex.from_arrays([["train"]*n_train + ["test"]*(len(data) - n_train), data.index.values], names = ["dataset", "date"])

df_forecast = pd.DataFrame(index = index,
                           columns = pd.MultiIndex.from_product([zones, ["nabqr"] + models, ("normal", "cdf", "original"), ("estimate", "lower prediction", "upper prediction")]),
                           dtype = np.float64
)

df_sim = pd.DataFrame(index = index,
                           columns = pd.MultiIndex.from_product([zones, ["nabqr"] + models, ("normal", "cdf", "original"), np.arange(N_sim)]),
                           dtype = np.float64
)
df_resids = pd.DataFrame(index = index,
                         columns = pd.MultiIndex.from_product([zones, ("normal", "cdf", "original")]),
                         dtype = np.float64
)

#%% nabqr

for par, val in zip(("estimate", "lower prediction", "upper prediction"), (0.5,0.05,0.95)):
    df_forecast.loc[idx[:,:], idx[:,"nabqr","normal", par]] = stats.norm().ppf(val)
    
df_sim.loc[idx[:,:], idx[:, "nabqr", "normal", :]] = stats.norm().rvs((len(data), len(zones) * N_sim))

#convert to original space
for zone in zones:
    actuals = data[zone,"Observed"].values.squeeze()
    est_quantiles = data[zone, "Quantiles"].values.squeeze()
    quant_est = qm.piecewise_linear_model(quantiles, *zone_lims[zone])
    
    tmp = quant_est.back_transform(est_quantiles, df_forecast.loc[idx[:,:], idx[zone,"nabqr","normal", :]])
    df_forecast.loc[idx[:,:], idx[zone,"nabqr","original", :]], df_forecast.loc[idx[:,:], idx[zone,"nabqr","cdf", :]] = tmp
    
    tmp = quant_est.back_transform(est_quantiles, df_sim.loc[idx[:,:], idx[zone,"nabqr","normal", :]])
    df_sim.loc[idx[:,:], idx[zone,"nabqr","original", :]], df_sim.loc[idx[:,:], idx[zone,"nabqr","cdf", :]] = tmp
    
    #also fill resid while we are at it
    tmp = quant_est.transform(est_quantiles, actuals)
    
    df_resids.loc[idx[:,:], idx[zone, "original"]] = actuals
    df_resids.loc[idx[:,:], idx[zone, "normal"]] = tmp[0].squeeze()
    df_resids.loc[idx[:,:], idx[zone, "cdf"]] = tmp[1].squeeze()
    
    #evaluate resids
    evaluation.evaluate_pseudoresids(df_resids.loc[idx["test",:], idx[zone, "cdf"]].values.squeeze()[N_step:],
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
        quant_est = qm.piecewise_linear_model(quantiles, *zone_lims[zone])
        
        normal, cdf, original, _ = correction(actuals, est_quantiles, quant_est,
                                               model_type = model,
                                               N_step = N_step, N_sim = N_sim,
                                               train_proportion=train_proportion,
                                               eval_resids= zone + "_" + str(model))
        
        df_forecast.loc[idx[:,:], idx[zone, model, "normal", :]] = normal[0]
        df_forecast.loc[idx[:,:], idx[zone, model, "cdf", :]] = cdf[0]
        df_forecast.loc[idx[:,:], idx[zone, model, "original", :]] = original[0]
        
        df_sim.loc[idx[:,:], idx[zone, model, "normal", :]] = normal[1]
        df_sim.loc[idx[:,:], idx[zone, model, "cdf", :]] = cdf[1]
        df_sim.loc[idx[:,:], idx[zone, model, "original", :]] = original[1]
        

#%% scores
df_scores = pd.DataFrame(index = ["nabqr"] + list(models),
                         columns = pd.MultiIndex.from_product([["train", "test"], zones, ["MAE", "MSE", "VARS", "CRPS"]]),
                         dtype = float)


for dataset in ["train", "test"]:
    print(dataset)
    
    for zone in zones:
        print(zone)
        #weights = evaluation.variogram_weight(df_sim.loc[idx[:,:], idx[zone, "nabqr", "original", :]].values)
        
        for p in df_scores.index:
            print(p)
            actuals = df_resids.loc[idx[dataset,:],idx[zone, "original"]].values.squeeze()[N_step:]
            predicted = df_forecast.loc[idx[dataset,:],idx[zone, p, "original", "estimate"]].values[N_step:]
            sim = df_sim.loc[idx[dataset,:], idx[zone, p, "original", :]].values[N_step:,:]
            
            #scores = evaluation.calc_scores(actuals, predicted, sim.values, VARS_kwargs = {"weights": weights})
            scores = evaluation.calc_scores(actuals, predicted, sim)
            
            df_scores.loc[p, idx[dataset, zone, :]] = scores


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
    x = df_forecast.index.get_level_values("date")
    
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
        index.get_level_values("date")[N_step-1::N_step],
        np.nanmin(df_forecast[zone].values),
        np.nanmax(df_forecast[zone].values),
        color = 'black',
        linestyle = '--'
    )
    ax.vlines(
        index.get_level_values("date")[n_train],
        np.nanmin(df_forecast[zone].values),
        np.nanmax(df_forecast[zone].values),
        color = 'black',
        linestyle = '-'
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

for dataset in ["train", "test"]:
    tmp = df_scores.xs(dataset, axis = 1).T.reset_index(names = ["zone", "metric"]).melt(
        value_vars = ["nabqr", "SARMA", "ARMA"],
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
    
    fig.savefig( (PATH / "Figures" / "Correlation Structure" / (dataset + "_scores.png" )))


#%% variogram

fig, axes = plt.subplots(len(zones), len(df_scores.index),
                         figsize = (12,14), layout = "constrained",
                         sharex = True, sharey = True)

c_range = []
for i, zone in enumerate(zones):
    for j, p in enumerate(df_scores.index):
        
        actuals = df_resids.loc[idx["test",:], idx[zone, "original"]].values[N_step:]
        sim = df_sim.loc[idx["test",:], idx[zone, p, "original", :]].values[N_step:,:]
        
        score, variogram = evaluation.variogram_score(sim, actuals)
        #variogram = np.log(variogram)
        np.fill_diagonal(variogram, 0)
        
        if p == "nabqr": c_range.append((variogram.min(),variogram.max()))
        
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