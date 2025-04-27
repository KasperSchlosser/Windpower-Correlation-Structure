import pathlib
import tomllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nabqra


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data"
save_path = PATH / "Results" / "Distribution"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)

load_path = PATH / "Data"
save_path = PATH / "Results" / "Distribution"

actuals = pd.read_pickle(load_path / "NABQR" / "actuals.pkl")
taqr_quantiles = pd.read_pickle(load_path / "NABQR" / "taqr_quantiles.pkl")
lstm_quantiles = pd.read_pickle(load_path / "NABQR" / "lstm_quantiles.pkl")

# ensure everythin has the same index
date_index = taqr_quantiles.index
actuals = actuals.loc[date_index]
taqr_quantiles = taqr_quantiles.loc[date_index]
lstm_quantiles = lstm_quantiles.loc[date_index]

zones = actuals.columns.get_level_values(0)
quantiles_str = lstm_quantiles["DK1-onshore"].columns.to_numpy()
quantiles = quantiles_str.astype(np.float64)


# %% fix the estimated quantiles
zone_limits = {
    "DK1-offshore": (0, 1300),
    "DK1-onshore": (0, 3600),
    "DK2-offshore": (0, 1100),
    "DK2-onshore": (0, 650)
}

for zone in zones:
    taqr_quantiles[zone] = nabqra.misc.fix_quantiles(taqr_quantiles[zone], *zone_limits[zone])
    lstm_quantiles[zone] = nabqra.misc.fix_quantiles(lstm_quantiles[zone], *zone_limits[zone])

# som of the observed values are less than 0
actuals = actuals.clip(lower=0.01)
# %%


lstm_cdf = pd.DataFrame(index=date_index,
                        columns=zones,
                        dtype=np.float64)
lstm_normal = pd.DataFrame(index=date_index,
                           columns=zones,
                           dtype=np.float64)
taqr_cdf = pd.DataFrame(index=date_index,
                        columns=zones,
                        dtype=np.float64)
taqr_normal = pd.DataFrame(index=date_index,
                           columns=zones,
                           dtype=np.float64)

for zone in zones:
    print(zone)
    qm = nabqra.quantiles.spline_model(quantiles,
                                       min_val=zone_limits[zone][0],
                                       max_val=zone_limits[zone][1]
                                       )
    print("taqr")
    tmp = qm.transform(taqr_quantiles[zone].values, actuals[zone].values)
    taqr_normal[zone] = tmp[0]
    taqr_cdf[zone] = tmp[1]
    print("lstm")
    tmp = qm.transform(lstm_quantiles[zone].values, actuals[zone].values)
    lstm_normal[zone] = tmp[0]
    lstm_cdf[zone] = tmp[1]


# %%
fig_kwargs = {"figsize": (14, 8), "layout": "tight"}
for zone in zones:
    nabqra.misc.pseudoresid_diagnostics(lstm_normal["DK1-onshore"], f'{zone}_lstm',
                                        color='navy',
                                        fig_kwargs=fig_kwargs,
                                        save_path=save_path / "Figures" / "Residuals")
    nabqra.misc.pseudoresid_diagnostics(lstm_normal["DK1-onshore"], f'{zone}_taqr',
                                        color='crimson',
                                        fig_kwargs=fig_kwargs,
                                        save_path=save_path / "Figures" / "Residuals")


# %% autocorrelation original space

fig_kwargs = {"figsize": (14, 8), "layout": "tight"}
for zone in zones:
    nabqra.misc.plot_autocorrelation(actuals[zone], f'{zone}_Original_ACF', estimator="ACF",
                                     color="navy",
                                     fig_kwargs=fig_kwargs,
                                     save_path=save_path / "Figures")
    nabqra.misc.plot_autocorrelation(actuals[zone], f'{zone}_Original_PACF', estimator="PACF",
                                     color="navy",
                                     fig_kwargs=fig_kwargs,
                                     save_path=save_path / "Figures")
