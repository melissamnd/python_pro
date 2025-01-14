import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

from . import risk_models

try:
    import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    raise ImportError("Please install matplotlib via pip or poetry")

def plot_covariance(cov_matrix, plot_correlation=False, show_tickers=True, **kwargs):

    # Convert covariance matrix to correlation if required
    matrix = cov_to_corr(cov_matrix) if plot_correlation else cov_matrix

    # Create the heatmap
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap='viridis')
    fig.colorbar(cax)

    # Configure tick labels
    if show_tickers:
        ax.set_xticks(np.arange(0, matrix.shape[0], 1))
        ax.set_xticklabels(matrix.index, rotation=90)
        ax.set_yticks(np.arange(0, matrix.shape[0], 1))
        ax.set_yticklabels(matrix.index)

    # Show the plot
    plt.show()

    return ax

def plot_weights(weights, tickers, ax=None, title="Portfolio Weights", **kwargs):
    """
    Plot portfolio weights as a horizontal bar chart.

    Parameters:
    - weights: 1D array of portfolio weights (e.g., [0.4, 0.3, 0.2, 0.1]).
    - tickers: List of asset tickers corresponding to the weights (e.g., ['AAPL', 'META']).
    - ax: Optional Matplotlib axis object.
    - title: Title of the plot (default: "Portfolio Weights").
    - **kwargs: Additional keyword arguments for customization.
    
    Returns:
    - ax: Matplotlib axis object.
    """
    ax = ax or plt.gca()  # Use provided axis or get the current one

    # Sort weights and tickers by weight (largest to smallest)
    desc = sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in desc]
    vals = [i[1] for i in desc]

    # Positions for the bars
    y_pos = np.arange(len(labels))

    # Create horizontal bar chart
    ax.barh(y_pos, vals, color=kwargs.get('color', 'blue'))
    ax.set_xlabel("Weight", fontsize=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()  # Invert y-axis for descending order
    ax.set_title(title, fontsize=14)

    # Display the plot
    plt.show()

    return ax


#Function to plot historical prices
def plot_historical_prices(df):
    """
    Plots historical prices from a DataFrame where tickers are columns,
    and the index represents dates.
    """
    tickers = df.columns 
    plt.figure(figsize=(12, 6))
    
    for ticker in tickers:
        plt.plot(df.index, df[ticker], label=ticker)
    
    plt.title("Historical Prices")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend()
    plt.show()

# Function to plot portfolio allocation as a pie chart
def plot_portfolio_allocation(portfolio):
    labels = list(portfolio.keys())
    sizes = list(portfolio.values())  
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Portfolio Allocation")
    plt.axis('equal')  
    plt.show()
