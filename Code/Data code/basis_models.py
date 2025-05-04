import pathlib
import tomllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import keras
import keras.ops as ops

from torch.utils.data import DataLoader

# from pandas import IndexSlice as idx

keras.utils.set_random_seed(666)


PATH = pathlib.Path.cwd().parents[1]
load_path = PATH / "Data"
save_path = PATH / "Data"

with open(PATH / "Settings" / "parameters.toml", "rb") as f:
    parameters = tomllib.load(f)
    zones = parameters["Zones"]
    zone_limits = parameters["Zone-Limits"]


# ensembles = pd.read_pickle(load_path / "Data" / "normalised_ensembles.pkl")
ensembles = pd.read_pickle(load_path / "Data" / "cleaned_ensembles.pkl")
ensembles = (ensembles - ensembles.groupby("Zone").mean()) / ensembles.groupby("Zone").std(ddof=1)
observations = pd.read_pickle(load_path / "Data" / "cleaned_observations.pkl")
observations /= observations.groupby("Zone").max()


class NABQRDataset(torch.utils.data.Dataset):

    def __init__(self, X, Y, timesteps=(0,), reverse=False):

        self.timesteps = torch.tensor(timesteps, dtype=torch.int)
        self.start = self.timesteps.max()

        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)

        self.length = self.Y.size(-1) - self.start

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
        x1 = keras.ops.multiply(self.quantiles, d)
        x2 = keras.ops.multiply(self.neg_quantiles, d)

        return ops.sum(ops.maximum(x1, x2), axis=-1)


quantiles = np.arange(0.05, 1.0, 0.05)
timesteps = np.array((0,))
# timesteps = np.arange(72)

# original_model = keras.Sequential(
#     [
#         keras.Input(shape=(len(timesteps), ensembles.shape[-1])),
#         keras.layers.Dropout(rate=0.3),
#         keras.layers.Dense(50, activation="celu"),
#         keras.layers.Dropout(rate=0.3),
#         keras.layers.Dense(20, activation="celu"),
#         keras.layers.LSTM(1, return_sequences=False),
#         keras.layers.Dense(20, activation="celu"),
#         keras.layers.Dense(50, activation="celu"),
#         keras.layers.Dense(len(quantiles)),
#     ]
# )

# original_model = keras.Sequential(
#     [
#         keras.Input(shape=(len(timesteps), ensembles.shape[-1])),
#         keras.layers.AveragePooling1D(52, data_format="channels_first"),
#         #        keras.layers.Dropout(rate=0.3),
#         keras.layers.Dense(50, activation="celu"),
#         keras.layers.Dropout(rate=0.3),
#         keras.layers.Dense(20, activation="celu"),
#         # keras.layers.LSTM(1, return_sequences=False),
#         # keras.layers.Dense(20, activation="celu"),
#         # keras.layers.Dense(50, activation="celu"),
#         keras.layers.Dense(len(quantiles)),
#     ]
# )

original_model = keras.Sequential(
    [
        keras.Input(shape=(len(timesteps), ensembles.shape[-1])),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(20, activation="celu"),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(3, activation="celu"),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(20, activation="celu"),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(len(quantiles)),
    ]
)

feature_extractor = keras.Model(inputs=original_model.inputs, outputs=original_model.layers[-4].output)


original_model.compile(
    loss=Quantileloss(quantiles),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
)

Loss = Quantileloss(quantiles)

models = {
    "Original - Relu": {
        "model": keras.Sequential(
            [
                keras.Input(shape=(len(timesteps), 52)),
                keras.layers.LSTM(256, return_sequences=False),
                keras.layers.Dense(len(quantiles), activation="sigmoid"),
                keras.layers.Dense(len(quantiles), activation="relu"),
            ]
        ),
        "compile": {"loss": Loss, "optimizer": keras.optimizers.Adam(learning_rate=1e-3)},
        "data": (ensembles, observations, (0, 1, 2, 5, 11, 23, 47)),
        "datamodel": NABQRDataset,
    },
    "Original - Identity": {
        "model": keras.Sequential(
            [
                keras.Input(shape=(len(timesteps), 52)),
                keras.layers.LSTM(256, return_sequences=False),
                keras.layers.Dense(len(quantiles), activation="sigmoid"),
                keras.layers.Dense(len(quantiles)),
            ]
        ),
        "compile": {"loss": Loss, "optimizer": keras.optimizers.Adam(learning_rate=1e-3)},
        "data": (ensembles, observations, (0, 1, 2, 5, 11, 23, 47)),
        "datamodel": NABQRDataset,
    },
}
# %%
training_size = int(0.64 * len(observations.index.unique(1)))
batch_size = 24 * 14


for zone in zones:

    train_data = NABQRDataset(
        ensembles.loc[zone].values[:training_size],
        observations.loc[zone].values[:training_size],
        timesteps=timesteps,
        reverse=True,
    )

    test_data = NABQRDataset(
        ensembles.loc[zone].values[training_size:],
        observations.loc[zone].values[training_size:],
        timesteps=timesteps,
        reverse=True,
    )

    history = original_model.fit(
        DataLoader(train_data, batch_size=batch_size),
        epochs=50,
        validation_data=DataLoader(test_data, batch_size=batch_size),
    )

    feature_extractor = keras.Model(inputs=original_model.inputs, outputs=original_model.layers[-4].output)

    preds = original_model.predict(DataLoader(test_data, batch_size=batch_size))
    preds = preds.squeeze()
    # preds2 = feature_extractor.predict(DataLoader(test_data, batch_size=batch_size))
    # fig, axes = plt.subplots(2, 1, sharex=True)
    # ax = axes[0]
    fig, ax = plt.subplots()
    ax.plot(test_data.Y[test_data.start :], color="black")
    ax.plot(preds[:, 9], color="blue")
    ax.plot(preds[:, [4, -5]], color="red")
    ax.plot(preds[:, [0, -1]], color="red", linestyle="--")

    # twin = ax.twinx()
    # twin.plot(-preds2, color="green")
    # ax = axes[1]
    # ax.plot(ensembles.loc[zone].values[training_size + timesteps[-1] :, 0], color="purple")
    # ax.plot(-preds2, color="green")
    ax.set_xlim([1300, 1700])
    ax.set_ylim([0, 1])
    break
