import pandas as pd
import numpy as np
import plotly.graph_objects as go

@dataclass
class PortfolioVisualizer_over_time:
    portfolio_history: list  # List of dictionaries with portfolio weights over time
    timestamps: list  # List of timestamps corresponding to portfolio weights

    def plot_portfolio_value(self, portfolio_values_df):
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