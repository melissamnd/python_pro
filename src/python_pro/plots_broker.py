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

        def plot_weights_over_time(self, portfolio_history_df):
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