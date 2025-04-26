import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class QuantileRegressionLSTM(tf.keras.Model):
    """LSTM-based model for quantile regression.
    Input: x -> LSTM -> Dense -> Dense -> output

    Parameters
    ----------
    n_quantiles : int
        Number of quantiles to predict
    units : int
        Number of LSTM units
    n_timesteps : int
        Number of time steps in input
    """

    def __init__(self, n_quantiles, units, n_timesteps, **kwargs):
        super().__init__(**kwargs)
        self.lstm = tf.keras.layers.LSTM(
            units, input_shape=(None, n_quantiles, n_timesteps), return_sequences=False
        )
        self.dense = tf.keras.layers.Dense(n_quantiles, activation="sigmoid")
        self.dense2 = tf.keras.layers.Dense(n_quantiles, activation="sigmoid")
        self.n_quantiles = n_quantiles
        self.n_timesteps = n_timesteps

    def call(self, inputs, training=None):
        """Forward pass of the model.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            Input tensor
        training : bool, optional
            Whether in training mode, by default None

        Returns
        -------
        tensorflow.Tensor
            Model output
        """
        x = self.lstm(inputs, training=training)
        x = self.dense(x)
        x = self.dense2(x)
        return x

    def get_config(self):
        """Get model configuration.

        Returns
        -------
        dict
            Model configuration
        """
        config = super(QuantileRegressionLSTM, self).get_config()
        config.update(
            {
                "n_quantiles": self.n_quantiles,
                "units": self.lstm.units,
                "n_timesteps": self.n_timesteps,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration.

        Parameters
        ----------
        config : dict
            Model configuration

        Returns
        -------
        QuantileRegressionLSTM
            Model instance
        """
        return cls(**config)


def train_model_lstm(
    quantiles,
    epochs: int, lr: float, batch_size: int,
    x, y, x_val, y_val,
    n_timesteps,
):
    """Train LSTM model for quantile regression.
    The @tf.function decorator is used to speed up the training process.


    Parameters
    ----------
    quantiles : list
        List of quantile levels to predict
    epochs : int
        Number of training epochs
    lr : float
        Learning rate for optimizer
    batch_size : int
        Batch size for training
    x : tensor
        Training input data
    y : tensor
        Training target data
    x_val : tensor
        Validation input data
    y_val : tensor
        Validation target data
    n_timesteps : int
        Number of time steps in input sequence

    Returns
    -------
    tf.keras.Model
        Trained LSTM model
    """
    model = QuantileRegressionLSTM(
        n_quantiles=len(quantiles), units=256, n_timesteps=n_timesteps
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            losses = quantile_loss_func(quantiles)(y_batch, y_pred)
            total_loss = tf.reduce_mean(losses)

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    @tf.function
    def val_step(x_batch, y_batch):
        y_pred = model(x_batch, training=False)
        losses = quantile_loss_func(quantiles)(y_batch, y_pred)
        total_loss = tf.reduce_mean(losses)
        return total_loss

    train_loss_history = []
    val_loss_history = []
    y_preds = []
    y_true = []

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        num_batches = 0

        # Training loop
        for i in range(0, len(x), batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            batch_train_loss = train_step(x_batch, y_batch)
            epoch_train_loss += batch_train_loss
            num_batches += 1

            y_preds.append(model(x_batch, training=False))
            y_true.append(y_batch)

        epoch_train_loss /= num_batches
        train_loss_history.append(epoch_train_loss)

        # Validation loop
        num_val_batches = 0
        for i in range(0, len(x_val), batch_size):
            x_val_batch = x_val[i : i + batch_size]
            y_val_batch = y_val[i : i + batch_size]

            batch_val_loss = val_step(x_val_batch, y_val_batch)
            epoch_val_loss += batch_val_loss
            num_val_batches += 1

        epoch_val_loss /= num_val_batches
        val_loss_history.append(epoch_val_loss)

        print(
            f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f} Validation Loss: {epoch_val_loss:.4f}"
        )

    return model
    

def remove_straight_line_outliers(ensembles):
    """Remove ensemble members that are perfectly straight lines (constant slope).
    Explanation: Sometimes the output from the LSTM is a straight line, which is not useful for the ensemble.

    Parameters
    ----------
    ensembles : numpy.ndarray
        2D array where rows are time steps and columns are ensemble members

    Returns
    -------
    numpy.ndarray
        Filtered ensemble data without straight-line outliers
    """
    # Calculate differences along the time axis
    differences = np.diff(ensembles, axis=0)

    # Identify columns where all differences are the same (perfectly straight lines)
    straight_line_mask = np.all(differences == differences[0, :], axis=0)

    # Remove the columns with perfectly straight lines
    return ensembles[:, ~straight_line_mask]


def remove_zero_columns(df):
    """Wrapper function to remove columns that contain only zeros from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame

    Returns
    -------
    pandas.DataFrame
        DataFrame with zero columns removed
    """
    return df.loc[:, (df != 0).any(axis=0)]


def remove_zero_columns_numpy(arr):
    """Remove columns that contain only zeros or constant values from a numpy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array

    Returns
    -------
    numpy.ndarray
        Array with zero/constant columns removed
    """
    return arr[:, (arr != 0).any(axis=0) & (arr != arr[0]).any(axis=0)]


def quantile_loss_3(q, y_true, y_pred):
    """Calculate quantile loss for a single quantile.

    Parameters
    ----------
    q : float
        Quantile level
    y_true : tensorflow.Tensor
        True values
    y_pred : tensorflow.Tensor
        Predicted values

    Returns
    -------
    tensorflow.Tensor
        Quantile loss value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tfp.stats.percentile(y_true, 100 * q, axis=1)
    error = y_true - y_pred
    return tf.maximum(q * error, (q - 1) * error)


def quantile_loss_func(quantiles):
    """Create a loss function for multiple quantiles.

    Parameters
    ----------
    quantiles : list
        List of quantile levels

    Returns
    -------
    function
        Loss function for multiple quantiles
    """

    def loss(y_true, y_pred):
        """Calculate the loss for given true and predicted values.

        Parameters
        ----------
        y_true : tensorflow.Tensor
            True values
        y_pred : tensorflow.Tensor
            Predicted values

        Returns
        -------
        tensorflow.Tensor
            Combined loss value for all quantiles
        """
        losses = []
        for i, q in enumerate(quantiles):
            loss = quantile_loss_3(q, y_true, y_pred[:, i])
            losses.append(loss)
        return tf.reduce_mean(tf.stack(losses))

    return loss
