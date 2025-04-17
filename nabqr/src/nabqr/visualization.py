import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

def legend_without_duplicate_labels(ax):
    """Create a legend without duplicate labels.
    Primarily used for ensemble plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to create legend for
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))


def visualize_results(y_hat, q_hat, ylabel):
    """Create a visualization of prediction intervals with actual values.

    Parameters
    ----------
    y_hat : numpy.ndarray
        Actual observed values
    q_hat : numpy.ndarray
        Predicted quantiles for different probability levels
    ylabel : str
        Label for the y-axis

    Returns
    -------
    None
        Saves the plot as 'TEST_NABQR_taqr_pi_plot.pdf' and displays it

    Notes
    -----
    - Creates a filled plot showing prediction intervals using a blue gradient
    - Overlays actual values as a black line
    - Automatically adjusts x-axis date formatting
    """
    y_hat = pd.Series(np.array(y_hat).flatten())
    try:
        taqr_results_corrected_plot = pd.DataFrame(np.array(q_hat).T, index=y_hat.index)
    except:
        taqr_results_corrected_plot = pd.DataFrame(np.array(q_hat), index=y_hat.index)

    m = taqr_results_corrected_plot.shape[1]  # Ensemble size
    # Define the color gradient from dark blue to light cyan
    colors = [
        (173 / 255, 217 / 255, 229 / 255),
        (19 / 255, 25 / 255, 148 / 255),
        (173 / 255, 217 / 255, 229 / 255),
    ]
    cmap = LinearSegmentedColormap.from_list("blue_to_cyan", colors, N=100)
    norm = plt.Normalize(vmin=0, vmax=m - 2)  # Normalize for the ensemble size
    sm = ScalarMappable(cmap=cmap, norm=norm)

    plt.figure(figsize=(6, 4))
    for i in range(m - 1):
        color = sm.to_rgba(i)
        plt.fill_between(
            taqr_results_corrected_plot.index,
            taqr_results_corrected_plot.iloc[:, i],
            taqr_results_corrected_plot.iloc[:, i + 1],
            color=color,
            alpha=1,
        )
    plt.plot(y_hat.index, y_hat, color="white", linewidth=3)  # White outline
    plt.plot(y_hat, color="black", label="Actuals", linewidth=1)  # Actual line
    plt.xlim(y_hat.index[0], y_hat.index[-1])

    # Create legend elements
    line = Line2D([0], [0], color="black", lw=2, label="Actuals")
    contour = Line2D(
        [0], [0], color=sm.to_rgba(m // 2), lw=5, alpha=0.9, label="Prediction Interval"
    )
    plt.legend(handles=[line, contour])

    plt.xlabel("Time")
    plt.ylabel(ylabel)

    # Configure date formatting on x-axis
    locator = mdates.AutoDateLocator(minticks=6, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(f"{ylabel}_taqr_pi_plot.pdf")
    plt.show()
