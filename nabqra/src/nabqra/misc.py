import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

from pandas import IndexSlice as idx


def fix_quantiles(quantiles: pd.DataFrame, min_val: float, max_val: float) -> pd.DataFrame:
    """
    Adjusts the quantile values within a DataFrame to ensure they fall within specified limits,
    performs linear interpolation, and returns the fixed quantiles.

    Parameters:
    -----------
    quantiles : pd.DataFrame
        columnes represent quantiles, while rows are each time step
    min_val : float
        The minimum value the data can take.
    max_val : float
        The maximum value the data can take.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with quantile values sorted, and forced into the specified interval

    Notes:
    ------
    For each row the values of the estimated quantiles are sorted.
    The values outside the specified interval are removed
    The missing values are filled by linear interpoltion to edges of interval
    """
    index = quantiles.index
    
    # Sort quantile values
    fixed_quantiles = quantiles.copy()
    fixed_quantiles.loc[idx[:], idx[:]] = np.sort(fixed_quantiles.to_numpy(), axis=1)
    
    # Replace all values outside of limits with NaNs
    mask = (fixed_quantiles > min_val) & (fixed_quantiles < max_val)
    fixed_quantiles = fixed_quantiles.where(mask, other=np.nan)
    
    # Add columns with min and max values for interpolation
    fixed_quantiles = pd.concat(
        (
            pd.Series(min_val, index=index),
            fixed_quantiles,
            pd.Series(max_val, index=index)
        ),
        axis=1
    )
    
    # Perform linear interpolation
    fixed_quantiles = fixed_quantiles.interpolate(axis=1)
    
    # Drop the extra columns
    fixed_quantiles = fixed_quantiles.drop(columns=fixed_quantiles.columns[[0, -1]])
    
    return fixed_quantiles


    
    index = quantiles.index
    
    #sort quantile values
    fixed_quantiles = quantiles.copy()
    fixed_quantiles.loc[idx[:], idx[:]] = np.sort(fixed_quantiles.to_numpy(), axis = 1)
    
    # replace all values outside of limits with nans
    mask = (fixed_quantiles > min_val) & (fixed_quantiles < max_val)
    fixed_quantiles = fixed_quantiles.where(mask, other = np.nan)
    
    #add cols with min and max values for interpolation
    fixed_quantiles = pd.concat((
            pd.Series(min_val, index = index),
            fixed_quantiles,
            pd.Series(max_val, index = index)
        ),
        axis = 1
    )
    
    #perform linear interpolation
    fixed_quantiles = fixed_quantiles.interpolate(axis = 1)
    
    #drop the extra columns
    fixed_quantiles = fixed_quantiles.drop(columns = fixed_quantiles.columns[[0,-1]])
    
    return fixed_quantiles


def plot_autocorrelation(values, name,
                         estimator = "ACF", alpha = 0.05, nlags = 48,
                         ax = None, fig_kwargs = None, color = "blue",
                         save_path = None):
    
    if fig_kwargs is None: fig_kwargs = dict()
    
    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)
    else:
        fig = ax.get_figure()
    
    match estimator:
        case "ACF":
            corr, interval = sm.tsa.acf(values, alpha = alpha, nlags = nlags)
        case "PACF": 
            corr, interval = sm.tsa.pacf(values, alpha = alpha, nlags = nlags)
            
    X = list(range(len(corr)))
    
    ax.bar(X, corr, color = color)
    ax.fill_between(X, interval[:,0] - corr, interval[:,1] - corr, color = color, alpha = 0.3)
    
    ax.set_title(name)
    ax.set_ylabel(estimator)
    ax.set_xlabel("Lag")
    
    if save_path is not None:
        fig.savefig(save_path / f'{name}.pdf', format = "pdf")
        plt.close(fig)
    
    return fig, ax

def pseudoresid_diagnostics(normal_resids: pd.DataFrame,
                            name,
                            alpha = 0.05, color = "blue",
                            num_points = 1000, nlags = 48,
                            fig_kwargs = None, save_path = None):
    
    if fig_kwargs is None: fig_kwargs = {}
    
    dist = stats.norm()
    
    index = normal_resids.index
    X = np.linspace(normal_resids.min(), normal_resids.max(), num_points)
    
    alpha_bonferroni = alpha / len(normal_resids)
    cdf_resids = dist.cdf(normal_resids)
    
    # cdf distribution
    fig, ax = plt.subplots(**fig_kwargs)
    
    ax.hist(cdf_resids, density = True, color = color)
    ax.hlines(1, 0, 1, colors = 'black', linestyles = 'dashed')
    ax.set_title(f'{name} distribution of CDF')
    
    if save_path is not None:
        fig.savefig(save_path / f'{name}_cdfdist.pdf', format = "pdf")
        plt.close(fig)
    
    # normal distribution
    fig, ax = plt.subplots(**fig_kwargs)
    ax.hist(normal_resids, density = True, color = color)
    ax.plot(X, dist.pdf(X), color = 'black', linestyle = 'dashed')
    ax.set_title(f'{name} distribution of normal residuals')
    
    if save_path is not None:
        fig.savefig(save_path / f'{name}_normaldist.pdf', format = "pdf")
        plt.close(fig)
    
    # outlier plot
    fig, ax = plt.subplots(**fig_kwargs)
    
    ax.scatter(index, normal_resids, color = color)
    ax.hlines(0, index[0], index[-1], color = "black")
    ax.hlines(dist.ppf([alpha / 2, 1 - alpha / 2]), index[0], index[-1], color = "green")
    ax.hlines(dist.ppf([alpha_bonferroni / 2, 1 - alpha_bonferroni / 2]), index[0], index[-1], color = "red")
    ax.set_title(f'{name} normal residuals')
    
    if save_path is not None:
        fig.savefig(save_path / f'{name}_outlier.pdf', format = "pdf")
        plt.close(fig)
    
    # qq-plot
    fig, ax = plt.subplots(**fig_kwargs)
    theo, obs = stats.probplot(normal_resids, dist = dist, fit = False)
    
    ax.scatter(theo,obs, color = color)
    ax.axline((0,0),slope = 1, color = 'black')
    ax.set_title(f'{name} normal QQ-plot')
    
    if save_path is not None:
        fig.savefig(save_path / f'{name}_qq.pdf', format = "pdf")
        plt.close(fig)
    
    # acf
    fig, ax = plot_autocorrelation(normal_resids, name + "_ACF",
                                   estimator="ACF",
                                   alpha = alpha, nlags = nlags,
                                   color=color, fig_kwargs = fig_kwargs,
                                   save_path=save_path)
    
    #pacf
    fig, ax = plot_autocorrelation(normal_resids, name + "_PACF",
                                   estimator="PACF",
                                   alpha = alpha, nlags = nlags,
                                   color=color, fig_kwargs = fig_kwargs,
                                   save_path=save_path)
    return
    
