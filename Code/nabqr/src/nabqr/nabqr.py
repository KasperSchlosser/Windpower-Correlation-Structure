"""Neural Adaptive Basis Quantile Regression (NABQR) Core Functions

This module provides the core functionality for NABQR.

This module includes:
- Scoring metrics (Variogram, CRPS, QSS)
- Dataset creation and preprocessing
- Model definitions and training
- TAQR (Time-Adaptive Quantile Regression) implementation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt

from .simulation import simulate_ar1, get_parameter_bounds, simulate_wind_power_sde
from .taqr import run_taqr
from .visualization import visualize_results
from .lstm import train_model_lstm, remove_zero_columns_numpy, remove_straight_line_outliers
from .scoring import calculate_scores


def create_dataset_for_lstm(X, Y, time_steps):
    """Create a dataset suitable for LSTM training with multiple time steps (i.e. lags)
    Parameters
    ----------
    X : numpy.ndarray
        Input features
    Y : numpy.ndarray
        Target values
    time_steps : list
        List of time steps to include
    Returns
    -------
    tuple
        (X_lstm, Y_lstm) LSTM-ready datasets
    """
    X = np.array(X)
    Y = np.array(Y)
    
    Xs, Ys = [], []
    for i in range(len(X)):
        X_entry = []
        for ts in time_steps:
            if i - ts >= 0:
                X_entry.append(X[i - ts, :])
            else:
                X_entry.append(np.zeros_like(X[0, :]))
        Xs.append(np.array(X_entry))
        Ys.append(Y[i])
    return np.array(Xs), np.array(Ys)


def pipeline(
    X, y,
    epochs = 100, training_size = 0.8, validation_size = 1000,
    taqr_init = "in-sample", quantiles_taqr = None, init_limit = 5000,
    quantiles_lstm = None, timesteps_lstm = None,
    names_lstm = None, names_taqr = None,
    save_name = None
):
    """Main pipeline for NABQR model training and evaluation.

    The pipeline:
    1. Trains an LSTM network to correct the provided ensembles
    2. Runs TAQR algorithm on corrected ensembles to predict observations
    3. Saves results and model artifacts

    Parameters
    ----------
    X : numpy.ndarray
        Shape (n_samples, n_features) - Ensemble data
    y : numpy.ndarray
        Shape (n_samples,) - Observations
    training_size : float, optional
        Fraction of data to use for training, by default 0.8
    epochs : int, optional
        Number of training epochs, by default 100
    timesteps_for_lstm : list, optional
        Time steps to use for LSTM input, by default [0, 1, 2, 6, 12, 24, 48]

    Returns
    -------
    tuple
        A tuple containing:
        - corrected_ensembles: pd.DataFrame
            The corrected ensemble predictions.
        - taqr_results: list of numpy.ndarray
            The TAQR results.
        - actuals_output: list of numpy.ndarray
            The actual output values.
        - BETA_output: list of numpy.ndarray
            The BETA parameters.
    """
    # Data preparation
    assert type(y) is pd.Series or type(y) is pd.DataFrame, "Observations y must be a Pandas Dataframe or Series"
    assert type(X) is pd.DataFrame, "Ensembles X Must Be a Pandas Dataframe "
    assert taqr_init in {"in-sample", "out-of-sample"}, "taqr_init must be one of In-sample, Out-of-Sample"
    
    if quantiles_taqr is None: quantiles_taqr = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    if quantiles_lstm is None: quantiles_lstm = np.arange(0.05,1,0.05)
    if timesteps_lstm is None: timesteps_lstm = np.array([0, 1, 2, 6, 12, 24, 48])
    if names_lstm is None: names_lstm = quantiles_lstm
    if names_taqr is None: names_taqr = quantiles_taqr
    
    idx = X.index.intersection(y.index)
    
    X = X.loc[idx,:]
    y = y.loc[idx,:]
    X_y = pd.concat((X, y), axis = 1)
    
    train_size = int(training_size * len(idx))
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]
    
    Xs, X_Ys = create_dataset_for_lstm(X, X_y, timesteps_lstm)

    # Handle NaN values
    if np.isnan(Xs).any():
        print("Xs has NaNs")
        Xs[np.isnan(Xs).any(axis=(1, 2))] = 0
    if np.isnan(X_Ys).any():
        print("X_Ys has NaNs")
        X_Ys[np.isnan(X_Ys).any(axis=1)] = 0

    # Data standardization
    min_val = np.min(X_Ys[:train_size])
    max_val = np.max(X_Ys[:train_size])
    def transformer(vals):
        return (vals - min_val) / (max_val - min_val)
    def detransformer(vals):
        return (max_val - min_val) * vals + min_val

    X_Ys_scaled = transformer(X_Ys)
    Xs_scaled = transformer(Xs)
    
    # Train LSTM model
    model = train_model_lstm(
        quantiles=quantiles_lstm,
        epochs=epochs,
        lr=1e-3,
        batch_size=50,
        x=tf.convert_to_tensor(Xs_scaled[:train_size]),
        y=tf.convert_to_tensor(X_Ys_scaled[:train_size]),
        x_val=tf.convert_to_tensor(Xs_scaled[train_size:train_size + validation_size]),
        y_val=tf.convert_to_tensor(X_Ys_scaled[train_size:train_size + validation_size]),
        n_timesteps=timesteps_lstm,
    )
    

    # run LSTM and sanitise LSTM output
    corrected_ensembles = model(tf.convert_to_tensor(Xs_scaled)).numpy()
    # i dont think these are need when running full
    #corrected_ensembles = remove_zero_columns_numpy(corrected_ensembles)
    #corrected_ensembles = remove_straight_line_outliers(corrected_ensembles)
    corrected_ensembles = pd.DataFrame(corrected_ensembles, index = idx, columns = names_lstm)
    # maybe these should be made in two steps
    # problem would be that the out-of-sample data get some information from the in-sample data
    # not comepletely analogues to when you would completely cold start on new data
    # but maybe not a big problem, when would you start on completely new data?
    
    # run taqr
    match taqr_init: 
        case "in-sample":
            n_init = len(train_idx) #i want all the test data
            n_full = len(corrected_ensembles)
            n_in_X = n_init
        
        case "out-of-sample":
            # we need enough data to initialise TAQR
            # but we also need data to validate on
            # take 25% of data if this is more than the limit
            # else take between 50% of the data or the limit
            corrected_ensembles = corrected_ensembles[test_idx]
            y = y[test_idx]
            
            n_full = len(y)
            n_init = max(int(0.25*len(n_full)), min(int(len(n_full) * 0.5), init_limit))
            n_in_X = n_full
            
            train_idx = test_idx[:n_init]
            test_idx = test_idx[n_init:]
    
    taqr_results, actuals_output, BETA_output = run_taqr(
        corrected_ensembles.values,
        y.values,
        quantiles_taqr,
        n_init,
        n_full,
        n_in_X
    )
    
    corrected_ensembles_original = detransformer(corrected_ensembles)

    training_results = {
        "Corrected ensembles": corrected_ensembles.loc[train_idx,:],
        "Corrected ensembles original space": corrected_ensembles_original.loc[train_idx,:],
        "Actuals": y.loc[train_idx,:],
        }
    
    test_results = {
        "Corrected ensembles": corrected_ensembles.loc[test_idx],
        "Corrected ensembles original space": corrected_ensembles_original.loc[test_idx],
        "Actuals": y.loc[test_idx],
        "Beta": pd.concat([pd.DataFrame(ar, index = test_idx[1:]) for ar in BETA_output], axis = 1, keys = names_taqr),
        "TAQR results": pd.DataFrame(np.vstack(taqr_results).T, index = test_idx[2:], columns = names_taqr),
        "model":model
        }
    
    
    if save_name:
        model.save(f'{save_name}.keras')
        
        for key, val in training_results.items():
            val.to_csv(f'{save_name} {key}.csv')
        for key, val in test_results.items():
            val.to_csv(f'{save_name} {key}.csv')

    return training_results, test_results


def run_nabqr_pipeline(
    n_samples=2000,
    phi=0.995,
    sigma=8,
    offset_start=10,
    offset_end=500,
    offset_step=15,
    correlation=0.8,
    data_source="NABQR-TEST",
    training_size=0.7,
    epochs=20,
    timesteps=[0, 1, 2, 6, 12, 24],
    quantiles=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    X=None,
    actuals=None,
    simulation_type="sde",
    visualize = True,
    taqr_limit=5000,
    save_files = True,
    ):
    """
    Run the complete NABQR pipeline, which may include data simulation, model training,
    and visualization. The user can either provide pre-computed inputs (X, actuals)
    or opt to simulate data if both are not provided.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of time steps to simulate if no data provided, by default 5000.
    phi : float, optional
        AR(1) coefficient for simulation, by default 0.995.
    sigma : float, optional
        Standard deviation of noise for simulation, by default 8.
    offset_start : int, optional
        Start value for offset range, by default 10.
    offset_end : int, optional
        End value for offset range, by default 500.
    offset_step : int, optional
        Step size for offset range, by default 15.
    correlation : float, optional
        Base correlation between dimensions, by default 0.8.
    data_source : str, optional
        Identifier for the data source, by default "NABQR-TEST".
    training_size : float, optional
        Proportion of data to use for training, by default 0.7.
    epochs : int, optional
        Number of epochs for model training, by default 100.
    timesteps : list, optional
        List of timesteps to use for LSTM, by default [0, 1, 2, 6, 12, 24].
    quantiles : list, optional
        List of quantiles to predict, by default [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99].
    X : array-like, optional
        Pre-computed input features. If not provided along with `actuals`, the function
        will prompt to simulate data.
    actuals : array-like, optional
        Pre-computed actual target values. If not provided along with `X`, the function
        will prompt to simulate data.
    simulation_type : str, optional
        Type of simulation to use, by default "ar1". "sde" is more advanced and uses a SDE model and realistic.
    visualize : bool, optional
        Determines if any visual elements will be plotted to the screen or saved as figures.
    taqr_limit : int, optional
        The lookback limit for the TAQR model, by default 5000.
    save_files : bool, optional
        Determines if any files will be saved, by default True. Note: the R-file needs to save some .csv files to run properly.
    Returns
    -------
    tuple
        A tuple containing:
            
        - corrected_ensembles: pd.DataFrame
            The corrected ensemble predictions.
        - taqr_results: list of numpy.ndarray
            The TAQR results.
        - actuals_output: list of numpy.ndarray
            The actual output values.
        - BETA_output: list of numpy.ndarray
            The BETA parameters.
        - scores: pd.DataFrame
            The scores for the predictions and original/corrected ensembles.
            
    Raises
    ------
    ValueError
        If user opts not to simulate data when both X and actuals are missing.
    """
    
    # If both X and actuals are not provided, ask user if they want to simulate
    if X is None or actuals is None:
        if X is not None or actuals is not None:
            raise ValueError("Either provide both X and actuals, or none at all.")
        choice = (
            input(
                "X and actuals are not provided. Do you want to simulate data? (y/n): "
            )
            .strip()
            .lower()
        )
        if choice != "y":
            raise ValueError(
                "Data was not provided and simulation not approved. Terminating function."
            )
            
        # Generate offset and correlation matrix for simulation
        offset = np.arange(offset_start, offset_end, offset_step)
        m = len(offset)
        corr_matrix = correlation * np.ones((m, m)) + (1 - correlation) * np.eye(m)
        
        # Generate simulated data
        # Check if simulation_type is valid
        if simulation_type not in ["ar1", "sde"]:
            raise ValueError("Invalid simulation type. Please choose 'ar1' or 'sde'.")
        if simulation_type == "ar1":    
            X, actuals = simulate_ar1(
                n_samples, phi, sigma, m, corr_matrix, offset, smooth=5
            )
        elif simulation_type == "sde":
            initial_params = {
                    'X0': 0.6,
                    'theta': 0.77,
                    'kappa': 0.12,        # Slower mean reversion
                    'sigma_base': 1.05,  # Lower base volatility
                    'alpha': 0.57,       # Lower ARCH effect
                    'beta': 1.2,        # High persistence
                    'lambda_jump': 0.045, # Fewer jumps
                    'jump_mu': 0.0,     # Negative jumps
                    'jump_sigma': 0.1    # Moderate jump size variation
                }
            # Check that initial parameters are within bounds
            bounds = get_parameter_bounds()
            for param, value in initial_params.items():
                lower_bound, upper_bound = bounds[param]
                if not (lower_bound <= value <= upper_bound):
                    print(f"Initial parameter {param}={value} is out of bounds ({lower_bound}, {upper_bound})")
                    if value < lower_bound:
                        initial_params[param] = lower_bound
                    else:
                        initial_params[param] = upper_bound
            
            t, actuals, X = simulate_wind_power_sde(
                initial_params, T=n_samples, dt=1.0
            )
            
        # Plot the simulated data with X in shades of blue and actuals in bold black
        plt.figure(figsize=(10, 6))
        cmap = plt.cm.Blues
        num_series = X.shape[1] if X.ndim > 1 else 1
        colors = [cmap(i) for i in np.linspace(0.3, 1, num_series)]  # Shades of blue
        if num_series > 1:
            for i in range(num_series):
                plt.plot(X[:, i], color=colors[i], alpha=0.7)
        else:
            plt.plot(X, color=colors[0], alpha=0.7)
        plt.plot(actuals, color="black", linewidth=2, label="Actuals")
        plt.title("Simulated Data")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
        
    # Run the pipeline
    corrected_ensembles, taqr_results, actuals_output, BETA_output, X_ensembles = pipeline(
        X,
        actuals,
        data_source,
        training_size=training_size,
        epochs=epochs,
        timesteps_for_lstm=timesteps,
        quantiles_taqr=quantiles,
        limit=taqr_limit,
        save_files = save_files
    )
    
    # Get today's date for file naming
    today = dt.datetime.today().strftime("%Y-%m-%d")
    
    # Visualize results
    if visualize:
        visualize_results(actuals_output, taqr_results, f"{data_source} example")
        
    # Calculate scores
    scores = calculate_scores(
        actuals_output,
        taqr_results,
        X_ensembles,
        corrected_ensembles,
        quantiles,
        data_source,
        plot_reliability=True,
        visualize = visualize
    )
    
    return corrected_ensembles, taqr_results, actuals_output, BETA_output, scores


if __name__ == "__main__":
    run_nabqr_pipeline()