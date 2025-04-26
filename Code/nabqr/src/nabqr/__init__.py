"""NABQR: Neural Adaptive Basis Quantile Regression

A method for sequential error-corrections tailored for wind power forecast in Denmark.
"""

from .nabqr import (
    pipeline,
    run_nabqr_pipeline,
)


from .lstm import (
    QuantileRegressionLSTM,
    train_model_lstm,
    quantile_loss_func,
    remove_zero_columns_numpy,
    remove_zero_columns,
)


from .scoring import (
    calculate_scores,
    reliability_func,
    quantile_score,
    variogram_score_single_observation,
    variogram_score_R_multivariate,
    calculate_crps,
    calculate_qss,
)


from .taqr import (
    run_taqr,
    set_n_closest_to_zero,
    set_n_smallest_to_zero,
    rq_simplex_final,
    one_step_quantile_prediction,
    opdatering_final,
    rq_initialiser_final,
    rq_simplex_alg_final,
    rq_purify_final,
)


from .simulation import (
    simulate_ar1,
    simulate_wind_power_sde,
    )


from .visualization import visualize_results

__all__ = [
    # Main functions
    "pipeline",
    "run_nabqr_pipeline",
    # Visualisation
    "visualize_results",
    #LSTM
    "QuantileRegressionLSTM",
    "train_model_lstm",
    #Scores
    "quantile_loss_func",
    "calculate_scores",
    "reliability_func",
    "quantile_score",
    "variogram_score_single_observation",
    "variogram_score_R_multivariate",
    "calculate_crps",
    "calculate_qss",
    #TAQR
    "run_taqr",
    "set_n_closest_to_zero",
    "set_n_smallest_to_zero",
    "rq_simplex_final",
    "one_step_quantile_prediction",
    "opdatering_final",
    "rq_initialiser_final",
    "rq_simplex_alg_final",
    "rq_purify_final",
]
