import pathlib
import tomllib

import pandas as pd
import numpy as np
import torch

import keras
import keras.ops as ops

from itertools import product
from torch.utils.data import DataLoader

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

    def __init__(self, X, Y, timesteps=(0,), reverse=False, normalisation=None):

        self.timesteps = torch.tensor(timesteps, dtype=torch.int)
        self.start = self.timesteps.max()

        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)

        self.length = self.Y.size(-1) - self.start

        match normalisation:
            case "MinMax":
                self.X = (self.X - self.X.min(axis=0)[0]) / (self.X.max(axis=0)[0] - self.X.min(axis=0)[0])
            case "Student":
                self.X = (self.X - self.X.mean(axis=0)[0]) / self.X.std(axis=0)[0]
            case None:
                pass

        if reverse:
            self.timesteps = torch.flip(self.timesteps, (-1,))

    def __len__(self):
        return int(self.length)

    def __getitem__(self, idx):
        return self.X[idx + self.start - self.timesteps, :], self.Y[idx + self.start]


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


# %% define models
ensembles = pd.read_pickle(load_path / "cleaned_ensembles.pkl")

observations = pd.read_pickle(load_path / "cleaned_observations.pkl")
obs_max = observations.groupby("Zone").max()
obs_scaled = observations / obs_max  # scale to help optimizer

test_index = ensembles.index.unique(level=1)[train_size:]

QUANTILES = np.arange(0.05, 1.0, 0.05)
quantiles_str = [f"{x:.2f}" for x in QUANTILES]
Loss = Quantileloss(QUANTILES)

models = [
    {
        "name": "Original - Relu",
        "model_config": keras.Sequential(
            [
                keras.Input(shape=(7, 52)),
                keras.layers.LSTM(256, return_sequences=False),
                keras.layers.Dense(len(QUANTILES), activation="sigmoid"),
                keras.layers.Dense(len(QUANTILES), activation="relu"),
            ]
        ).get_config(),
        "opt_args": {"learning_rate": 1e-3},
        "data_args": {"reverse": False, "timesteps": (0, 1, 2, 5, 11, 23, 47), "normalisation": "MinMax"},
        "loader_args": {"batch_size": 24 * 7},
        "fit_args": {"epochs": 150},
    },
    {
        "name": "Original - Identity",
        "model_config": keras.Sequential(
            [
                keras.Input(shape=(7, 52)),
                keras.layers.LSTM(256, return_sequences=False),
                keras.layers.Dense(len(QUANTILES), activation="sigmoid"),
                keras.layers.Dense(len(QUANTILES)),
            ]
        ).get_config(),
        "opt_args": {"learning_rate": 1e-3},
        "data_args": {"reverse": False, "timesteps": (0, 1, 2, 5, 11, 23, 47), "normalisation": "MinMax"},
        "loader_args": {"batch_size": 24 * 7},
        "fit_args": {"epochs": 100},
    },
    {
        "name": "Simple",
        "model_config": keras.Sequential(
            [
                keras.Input(shape=(1, ensembles.shape[-1])),
                keras.layers.Dropout(rate=0.5),
                keras.layers.Dense(25, activation="elu"),
                keras.layers.Dropout(rate=0.4),
                keras.layers.Dense(2),
                keras.layers.Dense(25, activation="elu"),
                keras.layers.Dropout(rate=0.2),
                keras.layers.Dense(len(QUANTILES)),
            ]
        ).get_config(),
        "opt_args": {"learning_rate": 1e-3},
        "data_args": {"reverse": True, "timesteps": (0,), "normalisation": "Student"},
        "loader_args": {"batch_size": 24 * 7, "shuffle": True},
        "fit_args": {"epochs": 100},
    },
]

# %% train and predict

history = {z: {} for z in zones}
predictions = {z: {} for z in zones}
features = pd.DataFrame(index=ensembles.index, columns=["Feature 1", "Feature 2"])


for zone, vals in product(zones, models):

    name = vals["name"]

    # for the test data  i need to pass in slightly more data
    # this is because the first index is where it is possible to get the  full data
    # ie if the models needs data from lag 48 the first data point is at 48
    # this just ensures all models test on the same data
    test_start = train_size - np.max(vals["data_args"]["timesteps"])

    train_data = NABQRDataset(
        ensembles.loc[zone].values[:train_size], obs_scaled.loc[zone].values[:train_size], **vals["data_args"]
    )
    train_data = DataLoader(train_data, **vals["loader_args"])

    test_data = NABQRDataset(
        ensembles.loc[zone].values[test_start:], obs_scaled.loc[zone].values[test_start:], **vals["data_args"]
    )
    test_data = DataLoader(test_data, **vals["loader_args"])

    model = keras.Sequential.from_config(vals["model_config"])
    model.compile(loss=Loss, optimizer=keras.optimizers.Adam(**vals["opt_args"]))

    hist = model.fit(train_data, validation_data=test_data, **vals["fit_args"])
    preds = model.predict(test_data)

    history[zone][name] = pd.DataFrame(hist.history)
    predictions[zone][name] = pd.DataFrame(preds.squeeze(), index=test_index, columns=quantiles_str)

    # i want to use the reduced space as features for regression
    if name == "Simple":
        extractor = keras.Model(inputs=model.inputs, outputs=model.layers[3].output)
        features.loc(axis=0)[zone, :] = extractor.predict(
            DataLoader(
                NABQRDataset(ensembles.loc[zone].values, obs_scaled.loc[zone].values, **vals["data_args"]),
                **vals["loader_args"],
            )
        ).squeeze()


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

history.to_csv(save_path / "history.csv")
history.to_pickle(save_path / "history.pkl")

predictions.to_csv(save_path / "Basis Normalised.csv")
predictions.to_pickle(save_path / "Basis Normalised.pkl")
predictions_original.to_csv(save_path / "Basis Quantiles.csv")
predictions_original.to_pickle(save_path / "Basis Quantiles.pkl")

features.to_csv(save_path / "Features.csv")
features.to_pickle(save_path / "Features.pkl")
