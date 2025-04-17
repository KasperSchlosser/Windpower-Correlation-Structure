import numpy as np
import properscoring as ps
import pandas as pd

def variogram_score_single_observation(x, y, p=0.5):
    """Calculate the Variogram score for a given observation.

    Translated from the R code in Energy and AI paper: 
    "An introduction to multivariate probabilistic forecast evaluation" by Mathias B.B. et al.

    Parameters
    ----------
    x : numpy.ndarray
        Ensemble forecast (m x k), where m is ensemble size, k is forecast horizon
    y : numpy.ndarray
        Actual observations (k,)
    p : float, optional
        Power parameter for the variogram score, by default 0.5

    Returns
    -------
    float
        Variogram score for the observation
    """
    m, k = x.shape
    score = 0

    for i in range(k - 1):
        for j in range(i + 1, k):
            Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j]) ** p)
            score += (1 / np.abs(i - j)) * (np.abs(y[i] - y[j]) ** p - Ediff) ** 2

    return score / k


def variogram_score_R_multivariate(x, y, p=0.5, t1=12, t2=36):
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
    m, k = x.shape
    score = 0
    if m > k:
        x = x.T
        m, k = k, m

    score_list = []
    for start in range(0, k, 24):
        if start + t2 <= k:
            for i in range(start + t1, start + t2 - 1):
                for j in range(i + 1, start + t2):
                    Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j]) ** p)
                    score += (1 / np.abs(i - j)) * (
                        np.abs(y[i] - y[j]) ** p - Ediff
                    ) ** 2
                score_list.append(score)

    return score / (100_000), score_list


def variogram_score_R_v2(x, y, p=0.5, t1=12, t2=36):
    """
    Calculate the Variogram score for all observations for the time horizon t1 to t2.
    Modified from the paper in Energy and AI, >> An introduction to multivariate probabilistic forecast evaluation <<.
    Assumes that x and y starts from day 0, 00:00.
    

    Parameters:
    x : array
        Ensemble forecast (m x k), where m is the size of the ensemble, and k is the maximal forecast horizon.
    y : array
        Actual observations (k,)
    p : float
        Power parameter for the variogram score.
    t1 : int
        Start of the hour range for comparison (inclusive).
    t2 : int
        End of the hour range for comparison (exclusive).

    Returns:
    --------
    tuple
        (score, score_list) Overall score/100_000 and list of individual VarS contributions
    """

    m, k = x.shape  # Size of ensemble, Maximal forecast horizon
    score = 0
    if m > k:
        x = x.T
        m, k = k, m
    else:
        print("m,k: ", m, k)

    score_list = []
    # Iterate through every 24-hour block
    for start in range(0, k, 24):
        # Ensure we don't exceed the forecast horizon
        if start + t2 <= k:
            for i in range(start + t1, start + t2 - 1):
                for j in range(i + 1, start + t2):
                    Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j]) ** p)
                    score += (1 / np.abs(i - j)) * (
                        np.abs(y[i] - y[j]) ** p - Ediff
                    ) ** 2
                score_list.append(score)

    # Variogram score
    return score / (100_000), score_list


def calculate_crps(actuals, corrected_ensembles):
    """Calculate the Continuous Ranked Probability Score (CRPS) using the properscoring package.
    If the ensembles do not have the correct dimensions, we transpose them.

    Parameters
    ----------
    actuals : numpy.ndarray
        Actual observations
    corrected_ensembles : numpy.ndarray
        Ensemble forecasts

    Returns
    -------
    float
        Mean CRPS score
    """
    try:
        crps = ps.crps_ensemble(actuals, corrected_ensembles)
        return np.mean(crps)
    except:
        crps = np.mean(ps.crps_ensemble(actuals, corrected_ensembles.T))
        return crps


def calculate_qss(actuals, taqr_results, quantiles):
    """Calculate the Quantile Skill Score (QSS).

    Parameters
    ----------
    actuals : numpy.ndarray
        Actual observations
    taqr_results : numpy.ndarray
        TAQR ensemble forecasts
    quantiles : array-like
        Quantile levels to evaluate

    Returns
    -------
    float
        Quantile Skill Score
    """
    qss_scores = multi_quantile_skill_score(actuals, taqr_results, quantiles)
    table = pd.DataFrame({
        "Quantiles": quantiles,
        "QSS NABQR": qss_scores
    })
    print(table)
    return np.mean(qss_scores)


def multi_quantile_skill_score(y_true, y_pred, quantiles):
    """Calculate the Quantile Skill Score (QSS) for multiple quantile forecasts.

    Parameters
    ----------
    y_true : numpy.ndarray
        True observed values
    y_pred : numpy.ndarray
        Predicted quantile values
    quantiles : list
        Quantile levels between 0 and 1

    Returns
    -------
    numpy.ndarray
        QSS for each quantile forecast
    """
    y_pred = np.array(y_pred)

    if y_pred.shape[0] > y_pred.shape[1]:
        y_pred = y_pred.T

    assert all(0 <= q <= 1 for q in quantiles), "All quantiles must be between 0 and 1"
    assert len(quantiles) == len(
        y_pred
    ), "Number of quantiles must match inner dimension of y_pred"

    N = len(y_true)
    scores = np.zeros(len(quantiles))

    for i, q in enumerate(quantiles):
        E = y_true - y_pred[i]
        scores[i] = np.sum(np.where(E > 0, q * E, (1 - q) * -E))

    return scores / N


def reliability_func(
    quantile_forecasts,
    corrected_ensembles,
    ensembles,
    actuals,
    corrected_taqr_quantiles,
    data_source,
    plot_reliability=True,
):
    n = len(actuals)

    # Handling hpe
    hpe = ensembles[:, 0]
    hpe_quantile = 0.5
    ensembles = ensembles[:, 1:]

    quantiles_ensembles = np.arange(2, 100, 2) / 100
    quantiles_corrected_ensembles = np.linspace(
        0.05, 0.95, corrected_ensembles.shape[1]
    ).round(3)

    # Ensuring that we are working with numpy arrays
    quantile_forecasts = (
        np.array(quantile_forecasts)
        if type(quantile_forecasts) != np.ndarray
        else quantile_forecasts
    )
    actuals = np.array(actuals) if type(actuals) != np.ndarray else actuals
    actuals_ensembles = actuals.copy()
    corrected_taqr_quantiles = (
        np.array(corrected_taqr_quantiles)
        if type(corrected_taqr_quantiles) != np.ndarray
        else corrected_taqr_quantiles
    )
    corrected_ensembles = (
        np.array(corrected_ensembles)
        if type(corrected_ensembles) != np.ndarray
        else corrected_ensembles
    )

    m, n1 = quantile_forecasts.shape
    if m != len(actuals):
        quantile_forecasts = quantile_forecasts.T
        m, n1 = quantile_forecasts.shape

    # Ensure that the length match up
    if len(actuals) != len(quantile_forecasts):
        if len(actuals) < len(quantile_forecasts):
            quantile_forecasts = quantile_forecasts[: len(actuals)]
        else:
            actuals_taqr = actuals[-len(quantile_forecasts) :]

    if len(actuals) != len(corrected_ensembles):
        if len(actuals) < len(corrected_ensembles):
            corrected_ensembles = corrected_ensembles[: len(actuals)]
        else:
            actuals_taqr = actuals[-len(corrected_ensembles) :]

    # Reliability: how often actuals are below the given quantiles compared to the quantile levels
    reliability_points_taqr = []
    for i, q in enumerate(corrected_taqr_quantiles):
        forecast = quantile_forecasts[:, i]
        observed_below = np.sum(actuals_taqr <= forecast) / n
        reliability_points_taqr.append(observed_below)

    reliability_points_taqr = np.array(reliability_points_taqr)

    reliability_points_ensembles = []
    n_ensembles = len(actuals_ensembles)
    for i, q in enumerate(quantiles_ensembles):
        forecast = ensembles[:, i]
        observed_below = np.sum(actuals_ensembles <= forecast) / n_ensembles
        reliability_points_ensembles.append(observed_below)

    reliability_points_corrected_ensembles = []
    for i, q in enumerate(quantiles_corrected_ensembles):
        forecast = corrected_ensembles[:, i]
        observed_below = np.sum(actuals_ensembles <= forecast) / n_ensembles
        reliability_points_corrected_ensembles.append(observed_below)

    # Handle hpe separately
    observed_below_hpe = np.sum(actuals_ensembles <= hpe) / n_ensembles

    reliability_points_ensembles = np.array(reliability_points_ensembles)


    # Plotting the reliability plot
    if plot_reliability:
        import scienceplots

        with plt.style.context("no-latex"):
            plt.figure(figsize=(6, 6))
            plt.plot(
                [0, 1], [0, 1], "k--", label="Perfect Reliability"
            )  # Diagonal line
            plt.scatter(
                corrected_taqr_quantiles,
                reliability_points_taqr,
                color="blue",
                label="Reliability CorrectedTAQR",
            )
            plt.scatter(
                quantiles_ensembles,
                reliability_points_ensembles,
                color="grey",
                label="Reliability Original Ensembles",
                marker="p",
                alpha=0.5,
            )
            plt.scatter(
                quantiles_corrected_ensembles,
                reliability_points_corrected_ensembles,
                color="green",
                label="Reliability Corrected Ensembles",
                marker="p",
                alpha=0.5,
            )
            plt.scatter(
                hpe_quantile,
                observed_below_hpe,
                color="grey",
                label="Reliability HPE",
                alpha=0.5,
                marker="D",
                s=25,
            )
            plt.xlabel("Nominal Quantiles")
            plt.ylabel("Observed Frequencies")
            plt.title(
                f'Reliability Plot for {data_source.replace("_", " ").replace("lstm", "")}'
            )
            plt.legend()
            plt.grid(True)
            plt.savefig(f"reliability_plots/nolatex_reliability_plot_{data_source}.pdf")
            plt.show()

    return (
        reliability_points_taqr,
        reliability_points_ensembles,
        reliability_points_corrected_ensembles,
    )


def calculate_scores(
    actuals,
    taqr_results,
    raw_ensembles,
    corrected_ensembles,
    quantiles_taqr,
    data_source,
    plot_reliability=True,
    visualize = True
):
    """Calculate Variogram, CRPS, QSS and MAE for the predictions and corrected ensembles.

    Parameters
    ----------
    actuals : numpy.ndarray
        The actual values
    predictions : numpy.ndarray
        The predicted values
    raw_ensembles : numpy.ndarray
        The raw ensembles
    corrected_ensembles : numpy.ndarray
        The corrected ensembles
    quantiles : list
        The quantiles to calculate the scores for
    data_source : str
        The data source
    """

    # Find common index
    common_index = corrected_ensembles.index.intersection(actuals.index)

    ensembles_CE_index = raw_ensembles.loc[common_index]
    actuals_comp = actuals.loc[common_index]

    variogram_score_raw_v2, _ = variogram_score_R_v2(
        ensembles_CE_index.loc[actuals_comp.index].values, actuals_comp.values
    )
    variogram_score_raw_corrected_v2, _ = variogram_score_R_v2(
        corrected_ensembles.loc[actuals_comp.index].values, actuals_comp.values
    )
    variogram_score_corrected_taqr_v2, _ = variogram_score_R_v2(
        taqr_results.values, actuals_comp.values
    )

    qs_raw = calculate_qss(
        actuals_comp.values,
        ensembles_CE_index.loc[actuals_comp.index].T,
        np.linspace(0.05, 0.95, ensembles_CE_index.shape[1]),
    )
    qs_corr = calculate_qss(
        actuals_comp.values,
        corrected_ensembles.loc[actuals_comp.index].T,
        np.linspace(0.05, 0.95, corrected_ensembles.shape[1]),
    )

    # TODO: Should be done with max and min from the training set. 
    taqr_values_clipped = np.clip(taqr_results, 0, max(actuals_comp.values))
    qs_corrected_taqr = calculate_qss(
        actuals_comp.values, taqr_values_clipped, quantiles_taqr
    )

    

    crps_orig_ensembles = calculate_crps(
        actuals_comp.values.flatten(), ensembles_CE_index.loc[actuals_comp.index].T
    )
    crps_corr_ensembles = calculate_crps(
        actuals_comp.values.flatten(), corrected_ensembles.loc[actuals_comp.index].T
    )
    crps_corrected_taqr = calculate_crps(
        actuals_comp.values.flatten(), np.array(taqr_results)
    )

    # Instead of calculating mean value of ensembles, we just use the median
    MAE_raw_ensembles = np.abs(
        np.median(ensembles_CE_index.loc[actuals_comp.index].values, axis=1)
        - actuals_comp.values
    )
    MAE_corr_ensembles = np.abs(
        np.median(corrected_ensembles.loc[actuals_comp.index].values, axis=1)
        - actuals_comp.values
    )
    MAE_corrected_taqr = np.abs(
        (np.median(np.array(taqr_results), axis=1) - actuals_comp.values)
    )

    scores_data = {
        "Metric": ["MAE", "CRPS", "Variogram", "QS"],
        "Original Ensembles": [
            np.mean(MAE_raw_ensembles),
            crps_orig_ensembles,
            variogram_score_raw_v2,
            np.mean(qs_raw),
        ],
        "Corrected Ensembles": [
            np.mean(MAE_corr_ensembles),
            crps_corr_ensembles,
            variogram_score_raw_corrected_v2,
            np.mean(qs_corr),
        ],
        "NABQR": [
            np.mean(MAE_corrected_taqr),
            crps_corrected_taqr,
            variogram_score_corrected_taqr_v2,
            np.mean(qs_corrected_taqr),
        ],
    }

    scores_df = pd.DataFrame(scores_data).T

    # Calculate relative scores
    scores_data["Corrected Ensembles"] = [
        1
        + (x - scores_data["Original Ensembles"][i])
        / scores_data["Original Ensembles"][i]
        for i, x in enumerate(scores_data["Corrected Ensembles"])
    ]
    scores_data["NABQR"] = [
        1
        + (x - scores_data["Original Ensembles"][i])
        / scores_data["Original Ensembles"][i]
        for i, x in enumerate(scores_data["NABQR"])
    ]

    # Create DataFrame
    scores_df = pd.DataFrame(scores_data).T

    print("Scores: ")
    print(scores_df)

    # Print LaTeX table
    latex_output = scores_df.to_latex(
        column_format="lcccc",
        header=True,
        float_format="%.3f",
        caption=f"Performance Metrics for Different Ensemble Methods on {data_source}",
        label="tab:performance_metrics",
        escape=False,
    ).replace("& 0 & 1 & 2 & 3 \\\\\n\\midrule\n", "")

    with open(f"latex_output_{data_source}_scores.tex", "w") as f:
        f.write(latex_output)

    # Reliability plot 
    reliability_func(
        taqr_results,
        corrected_ensembles,
        raw_ensembles,
        actuals,
        quantiles_taqr,
        data_source,
        plot_reliability = visualize,
    )

    return scores_df


def quantile_score(p, z, q):
    """Calculate the Quantile Score (QS) for a given probability and set of observations and quantiles.

    Implementation based on Fauer et al. (2021): "Flexible and consistent quantile estimation for
    intensity–duration–frequency curves"

    Parameters
    ----------
    p : float
        The probability level (between 0 and 1)
    z : numpy.ndarray
        The observed values
    q : numpy.ndarray
        The predicted quantiles

    Returns
    -------
    float
        The Quantile Score (QS)
    """
    u = z - q
    rho = np.where(u > 0, p * u, (p - 1) * u)
    return np.sum(rho)

