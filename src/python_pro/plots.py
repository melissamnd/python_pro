"""
  - ``plot_covariance`` - plot a correlation matrix
  - ``plot_weights`` - bar chart of weights
"""


import numpy as np
import scipy.cluster.hierarchy as sch

from . import risk_models

try:
    import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    raise ImportError("Please install matplotlib via pip or poetry")



def plot_covariance(cov_matrix, plot_correlation=False, show_tickers=True, **kwargs):
    if plot_correlation:
        matrix = risk_models.cov_to_corr(cov_matrix)
    else:
        matrix = cov_matrix
    fig, ax = plt.subplots()

    cax = ax.imshow(matrix)
    fig.colorbar(cax)

    if show_tickers:
        ax.set_xticks(np.arange(0, matrix.shape[0], 1))
        ax.set_xticklabels(matrix.index)
        ax.set_yticks(np.arange(0, matrix.shape[0], 1))
        ax.set_yticklabels(matrix.index)
        plt.xticks(rotation=90)

    _plot_io(**kwargs)

    return ax

def plot_weights(weights, ax=None, **kwargs):

    ax = ax or plt.gca()

    desc = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in desc]
    vals = [i[1] for i in desc]

    y_pos = np.arange(len(labels))

    ax.barh(y_pos, vals)
    ax.set_xlabel("Weight")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    _plot_io(**kwargs)
    return ax

# Function to plot historical prices
def plot_historical_prices(df):
    tickers = df['ticker'].unique()
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker]
        plt.plot(ticker_data['Date'], ticker_data['Close'], label=ticker)
    plt.title("Historical Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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
