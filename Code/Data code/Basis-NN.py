import pathlib
import tomllib

import pandas as pd
import numpy as np
import torch

import keras

from itertools import product
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from nabqra.scoring import Quantileloss

# from pandas import IndexSlice as idx

keras.utils.set_random_seed(666)

PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data" / "Data"
save_path = PATH / "Data" / "Basis"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]
    train_size = parameters["train_size"]


class NABQRDataset(torch.utils.data.Dataset):

    def __init__(self, X, Y, timesteps=(0,), reverse=False, preprocessing=None):

        self.timesteps = torch.tensor(timesteps, dtype=torch.int)
        self.start = self.timesteps.max()
        self.Y = torch.tensor(Y)

        if preprocessing is not None:
            X = preprocessing.transform(X)

        self.X = torch.tensor(X)

        if self.timesteps.max() > 0:
            pad = torch.zeros((self.timesteps.max(), X.shape[-1]))
            self.X = torch.cat((pad, self.X))

        self.start = self.timesteps.max()
        self.length = self.Y.size(-1)

        if reverse:
            self.timesteps = torch.flip(self.timesteps, (-1,))

    def __len__(self):
        return self.Y.size(-1)

    def __getitem__(self, idx):
        return self.X[idx + self.start - self.timesteps, :], self.Y[idx]


# %% define models
# load input, augment with additional data
ensembles = pd.read_pickle(load_path / "cleaned_ensembles.pkl")
# ensembles["Month-cos"] = np.cos(ensembles.index.get_level_values(1).month * 2 * np.pi / 12)
# ensembles["Month-sin"] = np.sin(ensembles.index.get_level_values(1).month * 2 * np.pi / 12)
# ensembles["Hour-cos"] = np.cos(ensembles.index.get_level_values(1).hour * 2 * np.pi / 24)
# ensembles["Hour-sin"] = np.sin(ensembles.index.get_level_values(1).hour * 2 * np.pi / 24)

observations = pd.read_pickle(load_path / "cleaned_observations.pkl")
obs_max = observations.groupby("Zone").max()
obs_scaled = observations / obs_max  # scale to help optimizer

date_index = ensembles.index.unique(level=1)

quantiles_str = [f"{x:.2f}" for x in parameters["Quantiles"]]
Loss = Quantileloss(parameters["Quantiles"])

models = [
    {
        "name": "Original - Relu",
        "model_config": keras.Sequential(
            [
                keras.Input(shape=(7, 52)),
                keras.layers.LSTM(256, return_sequences=False),
                keras.layers.Dense(len(parameters["Quantiles"]), activation="sigmoid"),
                keras.layers.Dense(len(parameters["Quantiles"]), activation="relu"),
            ]
        ).get_config(),
        "opt_args": {"learning_rate": 1e-3},
        "data_args": {"reverse": False, "timesteps": (0, 1, 2, 5, 11, 23, 47), "preprocessing": MinMaxScaler()},
        "loader_args": {"batch_size": 24 * 7},
        "fit_args": {"epochs": 2},
    },
    {
        "name": "Original - Identity",
        "model_config": keras.Sequential(
            [
                keras.Input(shape=(7, 52)),
                keras.layers.LSTM(256, return_sequences=False),
                keras.layers.Dense(len(parameters["Quantiles"]), activation="sigmoid"),
                keras.layers.Dense(len(parameters["Quantiles"])),
            ]
        ).get_config(),
        "opt_args": {"learning_rate": 1e-3},
        "data_args": {"reverse": False, "timesteps": (0, 1, 2, 5, 11, 23, 47), "preprocessing": MinMaxScaler()},
        "loader_args": {"batch_size": 24 * 7},
        "fit_args": {"epochs": 10},
    },
    {
        "name": "Simple",
        "model_config": keras.Sequential(
            [
                keras.Input(shape=(1, ensembles.shape[-1])),
                keras.layers.GaussianNoise(stddev=0.1),
                keras.layers.Dense(3),
                keras.layers.Dense(len(parameters["Quantiles"])),
            ]
        ).get_config(),
        "opt_args": {"learning_rate": 1e-3},
        "data_args": {"reverse": True, "timesteps": (0,), "preprocessing": StandardScaler()},
        "loader_args": {"batch_size": 12 * 1, "shuffle": False},
        "fit_args": {"epochs": 10},
    },
]

# %% train and predict

history = {z: {} for z in zones}
predictions = {z: {} for z in zones}
features = pd.DataFrame(index=ensembles.index, columns=["Feature 1", "Feature 2", "Feature 3"], dtype=np.float64)


for zone, vals in product(zones, models):

    name = vals["name"]
    print(zone, name)
    # for the test data  i need to pass in slightly more data
    # this is because the first index is where it is possible to get the  full data
    # ie if the models needs data from lag 48 the first data point is at 48
    # this just ensures all models test on the same data

    # inputs need to be preprocceseds
    if "preprocessing" in vals["data_args"]:
        vals["data_args"]["preprocessing"].fit(ensembles.loc[zone][train_size:].values)

    full_data = NABQRDataset(ensembles.loc[zone].values, obs_scaled.loc[zone].values, **vals["data_args"])
    full_data = DataLoader(full_data, **vals["loader_args"])

    train_data = NABQRDataset(
        ensembles.loc[zone].values[:train_size], obs_scaled.loc[zone].values[:train_size], **vals["data_args"]
    )
    train_data = DataLoader(train_data, **vals["loader_args"])

    test_data = NABQRDataset(
        ensembles.loc[zone].values[train_size:], obs_scaled.loc[zone].values[train_size:], **vals["data_args"]
    )
    test_data = DataLoader(test_data, **vals["loader_args"])

    model = keras.Sequential.from_config(vals["model_config"])
    model.compile(loss=Loss, optimizer=keras.optimizers.Adam(**vals["opt_args"]))

    hist = model.fit(train_data, validation_data=test_data, **vals["fit_args"])
    preds = model.predict(full_data)

    history[zone][name] = pd.DataFrame(hist.history)
    history[zone][name].index.name = "Epoch"

    predictions[zone][name] = pd.DataFrame(preds.squeeze(), index=date_index, columns=quantiles_str)

    # i want to use the reduced space as features for regression
    if name == "Simple":
        extractor = keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
        features.loc(axis=0)[zone, :] = extractor.predict(full_data).squeeze()


# %% save results
# maybe i should save the models?
history = pd.concat(
    {
        k: pd.concat(
            v,
            names=[
                "Model",
            ],
        )
        for k, v in history.items()
    },
    names=["Zone"],
)
predictions = pd.concat(
    {
        k: pd.concat(
            v,
            names=[
                "Model",
            ],
        )
        for k, v in predictions.items()
    },
    names=["Zone"],
)
predictions_original = predictions.mul(obs_max, axis=0)


# %%
history.to_csv(save_path / "history.csv")
history.to_pickle(save_path / "history.pkl")

predictions.to_csv(save_path / "Basis Normalised.csv")
predictions.to_pickle(save_path / "Basis Normalised.pkl")
predictions_original.to_csv(save_path / "Basis Quantiles.csv")
predictions_original.to_pickle(save_path / "Basis Quantiles.pkl")

features.to_csv(save_path / "Features.csv")
features.to_pickle(save_path / "Features.pkl")
