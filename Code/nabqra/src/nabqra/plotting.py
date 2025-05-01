import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import scipy.stats as stats


def plot_autocorrelation(values, name,
                         estimator="ACF", alpha=0.05, nlags=48,
                         ax=None, fig_kwargs=None,
                         save_path=None):

    if fig_kwargs is None:
        fig_kwargs = dict()

    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)
    else:
        fig = ax.get_figure()

    match estimator:
        case "ACF":
            stat = "ACF"
            corr, interval = sm.tsa.acf(values, alpha=alpha, nlags=nlags)
        case "PACF":
            stat = "PACF"
            corr, interval = sm.tsa.pacf(values, alpha=alpha, nlags=nlags)

    X = list(range(len(corr)))

    l = ax.bar(X, corr)
    ax.fill_between(X, interval[:, 0] - corr, interval[:, 1] - corr,
                    color=l.get_children()[0].get_facecolor(),
                    alpha=0.3)

    ax.set_title(name)
    ax.set_ylabel(estimator)
    ax.set_xlabel("Lag")

    if save_path is not None:
        fig.savefig(save_path / f'{name}_{stat}')
        plt.close(fig)
    return fig, ax


def pseudoresid_diagnostics(normal_resids, name,
                            alpha=0.05,
                            num_points=1000, nlags=72,
                            fig_kwargs=None, save_path=None):

    if fig_kwargs is None:
        fig_kwargs = {}

    dist = stats.norm()

    index = normal_resids.index
    X = np.linspace(normal_resids.min(), normal_resids.max(), num_points)

    alpha_bonferroni = alpha / len(normal_resids)
    cdf_resids = dist.cdf(normal_resids)

    # cdf distribution
    fig, ax = plt.subplots(**fig_kwargs)

    ax.hist(cdf_resids, density=True)
    ax.hlines(1, 0, 1, colors='black', linestyles='dashed')
    ax.set_title(name)
    ax.set_xlabel("u")
    ax.set_ylabel("Density")

    if save_path is not None:
        fig.savefig(save_path / f'{name}_cdfdist')
        plt.close(fig)

    # normal distribution
    fig, ax = plt.subplots(**fig_kwargs)
    ax.hist(normal_resids, density=True)
    ax.plot(X, dist.pdf(X), color='black', linestyle='dashed')
    ax.set_title(name)
    ax.set_xlabel("z")
    ax.set_ylabel("Density")

    if save_path is not None:
        fig.savefig(save_path / f'{name}_normaldist')
        plt.close(fig)

    # outlier plot
    fig, ax = plt.subplots(**fig_kwargs)

    ax.scatter(index, normal_resids)
    ax.hlines(0, index[0], index[-1], color="black")
    ax.hlines(dist.ppf([alpha / 2, 1 - alpha / 2]), index[0], index[-1], color="green")
    ax.hlines(dist.ppf([alpha_bonferroni / 2, 1 - alpha_bonferroni / 2]),
              index[0], index[-1], color="red")
    ax.set_title(name)
    ax.set_xlabel("Date")
    ax.set_ylabel("z")

    if save_path is not None:
        fig.savefig(save_path / f'{name}_outlier')
        plt.close(fig)

    # qq-plot
    fig, ax = plt.subplots(**fig_kwargs)
    theo, obs = stats.probplot(normal_resids, dist=dist, fit=False)

    ax.scatter(theo, obs)
    ax.axline((0, 0), slope=1, color='black')
    ax.set_title(name)
    ax.set_xlabel("Theoretical quantile")
    ax.set_ylabel("Observed quantile")

    if save_path is not None:
        fig.savefig(save_path / f'{name}_qq')
        plt.close(fig)

    # acf
    fig, ax = plot_autocorrelation(normal_resids, name,
                                   estimator="ACF",
                                   alpha=alpha, nlags=nlags,
                                   fig_kwargs=fig_kwargs,
                                   save_path=save_path)

    # pacf
    fig, ax = plot_autocorrelation(normal_resids, name,
                                   estimator="PACF",
                                   alpha=alpha, nlags=nlags,
                                   fig_kwargs=fig_kwargs,
                                   save_path=save_path)
    return


def multi_y_plot(X, Ys, ax=None,
                 labels=None, ylims=None,
                 offset=0.05, hide_original=True,
                 color_cycler=None,
                 label_pos="Legend"):

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
        ax.grid(False)

    offsets = [-offset*x for x in range(len(Ys))]

    lines = []
    for y, label, ylim, offset, color in zip(Ys, labels, ylims, offsets, color_cycler):

        color = color["color"]

        twin = ax.twinx()

        line = twin.plot(X, y, color=color)
        lines.append(line[0])

        twin.spines["left"].set_visible(True)
        twin.spines["left"].set_position(("axes", offset))
        twin.spines["left"].set_color(color)

        twin.yaxis.set_label_position('left')
        twin.yaxis.set_ticks_position('left')
        twin.tick_params(axis="y", which="both", colors=color, labelrotation=90)

        twin.set_ylim(ylim)

        twin.grid(False)

        if label_pos == "Axis":
            twin.set_ylabel(label, labelpad=0, color=color)
        else:
            twin.set_ylabel("", labelpad=0, color=color)

    if label_pos == "Legend":
        ax.legend(lines, labels)

    return fig, ax


def band_plot(x, y_mid, y1, y2=0, ax=None,
              color=None, alpha=0.3, label=None, band_label=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    p = ax.plot(x, y_mid, label=label, color=color)
    color = p[0].get_color()

    ax.fill_between(x, y1, y2, color=color, alpha=alpha, label=band_label)

    return fig, ax
