import pandas as pd
import numpy as np

from pandas import IndexSlice as idx


def fix_quantiles(quantiles: pd.DataFrame, min_val: float, max_val: float) -> pd.DataFrame:
    """
    Adjusts the quantile values within a DataFrame to ensure they fall within specified limits,
    performs linear interpolation, and returns the fixed quantiles.

    Parameters:
    -----------
    quantiles : pd.DataFrame
        columnes represent quantiles, while rows are each time step
    min_val : float
        The minimum value the data can take.
    max_val : float
        The maximum value the data can take.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with quantile values sorted, and forced into the specified interval

    Notes:
    ------
    For each row the values of the estimated quantiles are sorted.
    The values outside the specified interval are removed
    The missing values are filled by linear interpoltion to edges of interval
    """
    index = quantiles.index
    fixed_quantiles = quantiles.copy()

    # Replace all values outside of limits with NaNs
    mask = (fixed_quantiles > min_val) & (fixed_quantiles < max_val)
    fixed_quantiles = fixed_quantiles.where(mask, other=np.nan)

    # Add columns with min and max values for interpolation
    fixed_quantiles = pd.concat(
        (pd.Series(min_val, index=index), fixed_quantiles, pd.Series(max_val, index=index)), axis=1
    )

    # Perform linear interpolation
    fixed_quantiles = fixed_quantiles.interpolate(axis=1)

    # Drop the extra columns
    fixed_quantiles = fixed_quantiles.drop(columns=fixed_quantiles.columns[[0, -1]])

    # Sort quantile values
    fixed_quantiles.loc[idx[:], idx[:]] = np.sort(fixed_quantiles.to_numpy(), axis=1)

    return fixed_quantiles
