import numpy as np
import scipy.integrate as integrate
import properscoring as ps
import keras
import torch
from keras import ops


def variogram_weight(simulations, p=0.5, window=24, offset=24):

    n, m = simulations.shape
    n_windows = (n - offset) // window

    weights = np.zeros((n_windows, window, window))

    for k, start in enumerate(range(offset, n, window)):

        if start + window > n:
            break

        for i in range(0, window - 1):
            for j in range(i + 1, window):

                Ediff = np.abs(simulations[start + i, :] - simulations[start + j, :]) ** p

                weights[k, i, j] = 1 / np.var(Ediff, ddof=1)
    return weights


def variogram_score(simulations, actuals, p=0.5, window=24, offset=24, weights=None):

    # simualtions are n observations times m simulations
    n, m = simulations.shape

    score = 0
    variogram = np.zeros((window, window))
    n_windows = (n - offset) // window

    if weights is None:
        weights = np.array([[abs(i - j) for i in range(0, window)] for j in range(0, window)])
        np.fill_diagonal(weights, 1)  # just to not get warning, is not used
        weights = 1 / weights
        weights = np.repeat(
            weights[
                np.newaxis,
                :,
                :,
            ],
            n_windows,
            axis=0,
        )

    for k, start in enumerate(range(offset, n, window)):

        if start + window > n:
            break

        for i in range(0, window - 1):
            for j in range(i + 1, window):
                Ediff = np.abs(simulations[start + i, :] - simulations[start + j, :]) ** p
                Adiff = np.abs(actuals[start + i] - actuals[start + j]) ** p

                s = weights[k, i, j] * (Adiff - np.mean(Ediff)) ** 2
                variogram[i, j] += s
                variogram[j, i] += s
                score += s

    return score / n_windows, variogram / n_windows


def continous_ranked_probability_score(simulations, actuals):
    # n observation times k simulations
    return np.mean(ps.crps_ensemble(actuals, simulations))


def mean_average_error(actuals, predicted):
    return np.mean(np.abs(actuals - predicted))


def mean_squared_error(actuals, predicted):
    return np.mean((actuals - predicted) ** 2)


class Quantileloss(keras.losses.Loss):
    def __init__(self, quantiles, **kwargs):
        super().__init__(**kwargs)
        self.quantiles = torch.tensor(quantiles)
        self.neg_quantiles = self.quantiles - 1

    def call(self, y_true, y_pred):
        d = (y_true - y_pred.T).T
        x1 = self.quantiles * d
        x2 = self.neg_quantiles * d

        return ops.sum(ops.maximum(x1, x2), axis=-1)


def calc_scores(actuals, predicted, simulations, VARS_kwargs=None):

    # maybe return dict?

    if VARS_kwargs is None:
        VARS_kwargs = dict()

    MAE = mean_average_error(actuals, predicted)
    MSE = mean_squared_error(actuals, predicted)
    VARS = variogram_score(simulations, actuals, **VARS_kwargs)[0]
    CRPS = continous_ranked_probability_score(simulations, actuals)

    return MAE, MSE, VARS, CRPS


def continous_wasserstein(F_inv, G_inv, lims=(1e-8, 1 - 1e-8), order=1, **kwargs):
    # quick implementation of the wasserstein metric
    # F_inf, G_inv: quantile (inverse cdf) functions for the distributions F, G
    # lim: limits of integration, change if distribution has infinite support

    def _dist(x):
        return np.abs(F_inv(x) - G_inv(x)) ** order

    res = integrate.quad(_dist, *lims, **kwargs)

    return (res[0] ** (1 / order), *res[1:])


def continous_kl_divergence(f, g, lims=(-np.inf, np.inf), **kwargs):

    def _div(x):
        fx = f(x)
        gx = g(x)
        return fx * (np.log(fx) - np.log(gx))

    return integrate.quad(_div, *lims, **kwargs)
