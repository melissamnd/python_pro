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