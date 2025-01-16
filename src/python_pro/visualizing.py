import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from .risk_models import cov_to_corr

#---------------------------------------------------------
# Graphic analysis of the generated backtest file.
#---------------------------------------------------------

class PortfolioVisualizer:
    def __init__(self, data=None):
        """
        Constructor to initialize the PortfolioVisualizer class.

        Parameters:
        - data: Optional DataFrame containing asset data (prices, weights, etc.).
        """
        self.data = data

    def plot_covariance(self, cov_matrix, plot_correlation=False, show_tickers=True, **kwargs):
        """
        Plot the covariance (or correlation) matrix as a heatmap.
        
        Parameters:
        - cov_matrix: Covariance matrix to be plotted.
        - plot_correlation: Whether to plot correlation instead of covariance (default: False).
        - show_tickers: Whether to display tickers as labels (default: True).
        - **kwargs: Additional arguments passed to imshow and customization.
        """
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

    def plot_weights(self, weights, tickers, ax=None, title="Portfolio Weights", **kwargs):
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

    def plot_historical_prices(self, df):
        """
        Plots historical prices from a DataFrame where tickers are columns,
        and the index represents dates.

        Parameters:
        - df: DataFrame containing historical prices of assets.
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

    def plot_portfolio_allocation(self, portfolio):
        """
        Plot the portfolio allocation as a pie chart.

        Parameters:
        - portfolio: Dictionary with asset tickers as keys and weights as values.
        """
        labels = list(portfolio.keys())
        sizes = list(portfolio.values())  
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title("Portfolio Allocation")
        plt.axis('equal')  
        plt.show()

    def plot_portfolio_value(self, portfolio_values_df):
        """
        Plot portfolio value over time.

        Parameters:
        - portfolio_values_df: DataFrame containing portfolio values over time.
        """
        fig = go.Figure(data=go.Scatter(
            x=portfolio_values_df.index,
            y=portfolio_values_df['Portfolio Value'],
            mode='lines',
            name='Portfolio Value'
        ))

        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            xaxis=dict(tickformat="%Y-%m-%d"),
            yaxis=dict(tickprefix="$"),
            width=900,
            height=500,
        )

        fig.show()

    def plot_cumulative_performance(self, portfolio_returns, benchmark_returns):
        """
        Plot cumulative portfolio performance against a benchmark.

        Parameters:
        - portfolio_returns: DataFrame of portfolio returns over time.
        - benchmark_returns: DataFrame of benchmark returns over time.
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=portfolio_returns.index,
            y=(1 + portfolio_returns).cumprod(),
            mode='lines',
            name='Portfolio Cumulative Return'
        ))

        fig.add_trace(go.Scatter(
            x=benchmark_returns.index,
            y=(1 + benchmark_returns).cumprod(),
            mode='lines',
            name='Benchmark Cumulative Return'
        ))

        fig.update_layout(
            title="Cumulative Portfolio Performance vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            xaxis=dict(tickformat="%Y-%m-%d"),
            yaxis=dict(tickformat="%"),
            width=900,
            height=500,
        )

        fig.show()

    def plot_weights_over_time(self, portfolio_history_df):
        """
        Plot the weights of assets in the portfolio over time.

        Parameters:
        - portfolio_history_df: DataFrame containing portfolio weights over time.
        """
        fig = go.Figure()

        for asset in portfolio_history_df.columns:
            fig.add_trace(go.Scatter(
                x=portfolio_history_df.index,
                y=portfolio_history_df[asset],
                mode='lines',
                stackgroup='one',
                name=asset
            ))

        fig.update_layout(
            title="Portfolio Weights Over Time",
            xaxis_title="Date",
            yaxis_title="Weight",
            xaxis=dict(tickformat="%Y-%m-%d"),
            yaxis=dict(tickformat=".0%"),
            width=900,
            height=500,
        )

        fig.show()

    def plot_drawdown(self, portfolio_returns):
        """
        Plot the drawdown of the portfolio.

        Parameters:
        - portfolio_returns: DataFrame containing portfolio returns over time.
        """
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak

        fig = go.Figure(data=[
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name="Drawdown"
            )
        ])

        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            xaxis=dict(tickformat="%Y-%m-%d"),
            yaxis=dict(tickformat=".0%"),
            width=900,
            height=500,
        )

        fig.show()

