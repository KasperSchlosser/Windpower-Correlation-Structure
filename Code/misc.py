import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

def evaluate_pseudoresids(pseudo_resids, index = None, save_path = None, name = None, figsize = (14,8), close_figs = False):
    #functions to visually evaluate the pseudo residuals
    
    if index is None:
        index = list(range(len(pseudo_resids)))
    
    if name is None:
        name = "psudo_residual"
    
    dist = stats.norm()
    resids = dist.ppf(pseudo_resids)
    
    theo_quantiles = dist.ppf(np.arange(1,len(pseudo_resids)+1) / (len(pseudo_resids)+1))
    
    figs = [plt.figure(figsize = figsize, layout = "tight") for _ in range(5)]
    
    # Scatter plot
    ax = figs[0].subplots()
    sns.scatterplot(x = index,  y = resids, ax = ax)
    
    lims = (dist.ppf(0.05 / (2*len(pseudo_resids))), dist.ppf(1 - 0.05 / (2*len(pseudo_resids))))
    ax.hlines(0, index[0], index[-1], color = 'black')
    ax.hlines((-2,2), index[0], index[-1], color = 'navy', linestyle = 'dashed')
    ax.hlines(lims, index[0], index[-1], color = "crimson")
    
    if save_path is not None:
        figs[0].savefig( save_path / (name + "_scatter.png" ))
        
    # Uniform histogram
    ax = figs[1].subplots()
    sns.histplot(x = pseudo_resids, stat = "density", ax = ax)
    ax.hlines(1,0,1, color = 'black')
    
    if save_path is not None:
        figs[1].savefig( save_path / (name + "_cdfdist.png" ))
    
    
    #normal histogram
    ax = figs[2].subplots()
    
    sns.histplot(x = resids, stat = "density", ax = ax)
    sns.lineplot(x = theo_quantiles, y = dist.pdf(theo_quantiles), ax = ax, color = 'black')
    if save_path is not None:
        figs[2].savefig( save_path / (name + "_normaldist.png" ))
    
    # propplot
    ax = figs[3].subplots()
    
    sns.scatterplot(x = theo_quantiles, y = sorted(resids), ax = ax)
    ax.axline((theo_quantiles[0], theo_quantiles[0]), (theo_quantiles[-1], theo_quantiles[-1]))
    ax.set_xlabel("Theoretical Quantile")
    ax.set_ylabel("Observed Quantile")
    
    if save_path is not None:
        figs[3].savefig( save_path / (name + "_probplot.png" ))
    
    # (pacf) plot
    axs = figs[4].subplots(1,2)
    
    acf_vals, conf_acf = sm.tsa.acf(resids, alpha = 0.05)
    pacf_vals, conf_pacf = sm.tsa.pacf(resids, alpha = 0.05)
    
    sns.barplot(x = np.arange(len(acf_vals)), y = acf_vals, ax = axs[0])
    axs[0].fill_between(
        np.arange(len(acf_vals)),
        conf_acf[:,0] - acf_vals,
        conf_acf[:,1] - acf_vals,
        color = 'black',
        alpha = 0.3
    )
    axs[0].set_title("ACF")
    
    sns.barplot(x = np.arange(len(pacf_vals)), y = pacf_vals, ax = axs[1])
    axs[1].fill_between(
        np.arange(len(pacf_vals)),
        conf_pacf[:,0] - pacf_vals,
        conf_pacf[:,1] - pacf_vals,
        color = 'black',
        alpha = 0.3
    )
    axs[1].set_title("PACF")
    
    if save_path is not None:
        figs[4].savefig( save_path / (name + "_autocorrelation.png" ))
        
    if close_figs:
        for fig in figs:
            plt.close(fig)
        return
    
    return figs