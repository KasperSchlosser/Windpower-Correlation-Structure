import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import properscoring as ps
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

def variogram_weight(simulations,
                    p=0.5, window = 24, offset = 24):
    
    n,m = simulations.shape
    n_windows = (n - offset) // window 
    
    weights = np.zeros((n_windows, window, window))
    
    for k,start in enumerate(range(offset, n, window)):
        
        if start + window > n: break
    
        for i in range(0,window-1):
            for j in range(i+1, window):
                
                Ediff = np.abs(simulations[start + i, :] - simulations[start + j, :]) ** p
                
                weights[k,i,j] = 1 / np.var(Ediff, ddof = 1)
    return weights
    
    
def variogram_score(simulations, actuals,
                    p=0.5, window = 24, offset = 24,
                    weights = None):
    
    #simualtions are n observations times m simulations
    n,m = simulations.shape
    
    score = 0
    variogram = np.zeros((window,window))
    n_windows = (n - offset) // window
    
    if weights is None:
        weights = np.array([[abs(i-j) for i in range(0,window)] for j in range(0,window)])
        np.fill_diagonal(weights, 1) # just to not get warning, is not used
        weights = 1/weights
        weights = np.repeat(weights[np.newaxis,:,:,], n_windows, axis = 0)
    
    for k,start in enumerate(range(offset, n, window)):
        
        if start + window > n: break
        
        for i in range(0,window-1):
            for j in range(i+1, window):
                Ediff =  np.abs(simulations[start + i, :] - simulations[start + j, :]) ** p
                Adiff = np.abs(actuals[start + i] - actuals[start + j]) ** p
                
                
                s = weights[k,i,j] * (Adiff - np.mean(Ediff)) ** 2
                variogram[i,j] += s
                variogram[j,i] += s
                score += s


    return score / n_windows, variogram / n_windows

def continous_ranked_probability_score(simulations, actuals):
    #n observation times k simulations
    return np.mean(ps.crps_ensemble(actuals, simulations))

def mean_average_error(actuals, predicted):
    return np.mean(np.abs(actuals - predicted))

def mean_squared_error(actuals, predicted):
    return np.mean((actuals - predicted)**2)

def calc_scores(actuals, predicted, simulations, VARS_kwargs = None):
    
    #maybe return dict?
    
    if VARS_kwargs is None:
        VARS_kwargs = dict()
    
    MAE = mean_average_error(actuals, predicted)
    MSE = mean_squared_error(actuals, predicted)
    VARS = variogram_score(simulations, actuals, **VARS_kwargs)[0]
    CRPS = continous_ranked_probability_score(simulations, actuals)
    
    return MAE, MSE, VARS, CRPS
