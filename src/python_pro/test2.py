import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from pybacktestchain.data_module import UNIVERSE_SEC, FirstTwoMoments, get_stocks_data, DataModule, Information
from pybacktestchain.utils import generate_random_name
from pybacktestchain.blockchain import Block, Blockchain

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#---------------------------------------------------------
# Classes
#---------------------------------------------------------

@dataclass
class Backtest:
    initial_date: datetime
    final_date: datetime
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'NFLX']
    information_class: type = Information
    s: timedelta = timedelta(days=360)
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column: str = 'Adj Close'
    rebalance_flag: type = EndOfMonth
    risk_model: type = StopLoss
    initial_cash: int = 1000000  # Initial cash in the portfolio
    name_blockchain: str = 'backtest'
    verbose: bool = True
    broker = Broker(cash=initial_cash, verbose=verbose)

    def __post_init__(self):
        self.backtest_name = generate_random_name()
        self.broker.initialize_blockchain(self.name_blockchain)

    def run_backtest(self):
        logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")
        df = get_stocks_data(self.universe, self.initial_date.strftime('%Y-%m-%d'), self.final_date.strftime('%Y-%m-%d'))

        # Initialize the DataModule
        data_module = DataModule(df)

        # Create the Information object
        info = self.information_class(
            s=self.s, 
            data_module=data_module,
            time_column=self.time_column,
            company_column=self.company_column,
            adj_close_column=self.adj_close_column
        )

        # Run the backtest
        for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):
            if self.risk_model is not None:
                portfolio = info.compute_portfolio(t, info.compute_information(t))
                prices = info.get_prices(t)
                self.risk_model.trigger_stop_loss(t, portfolio, prices, self.broker)

            if self.rebalance_flag().time_to_rebalance(t):
                logging.info(f"Rebalancing portfolio at {t}")
                information_set = info.compute_information(t)
                portfolio = info.compute_portfolio(t, information_set)
                prices = info.get_prices(t)
                self.broker.execute_portfolio(portfolio, prices, t)

        final_portfolio_value = self.broker.get_portfolio_value(info.get_prices(self.final_date))
        logging.info(f"Backtest completed. Final portfolio value: {final_portfolio_value}")
        df = self.broker.get_transaction_log()

        # Save results
        if not os.path.exists('backtests'):
            os.makedirs('backtests')
        df.to_csv(f"backtests/{self.backtest_name}.csv")

        # Store results in the blockchain
        self.broker.blockchain.add_block(self.backtest_name, df.to_string())

        return df

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.01):
        """
        Calculate the Sharpe Ratio of the portfolio.
        Args:
            returns (pd.Series): Daily portfolio returns.
            risk_free_rate (float): Annual risk-free rate, default is 1%.
        Returns:
            float: Sharpe Ratio.
        """
        excess_returns = returns - (risk_free_rate / 252)  # Convert annual risk-free rate to daily
        sharpe_ratio = excess_returns.mean() / excess_returns.std()
        sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)  # Annualize Sharpe Ratio
        return sharpe_ratio_annualized

    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95):
        """
        Calculate the Value at Risk (VaR) of the portfolio.
        Args:
            returns (pd.Series): Daily portfolio returns.
            confidence_level (float): Confidence level for VaR, default is 95%.
        Returns:
            float: Value at Risk.
        """
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var

    def plot_correlation_heatmap(self, prices: pd.DataFrame):
        """
        Plot a correlation heatmap of the asset returns.
        Args:
            prices (pd.DataFrame): DataFrame of asset prices.
        """
        returns = prices.pct_change().dropna()
        correlation_matrix = returns.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
        plt.title("Correlation Heatmap of Asset Returns")
        plt.show()

    def analyze_results(self, prices_history: pd.DataFrame):
        """
        Perform analysis on the backtest results, including Sharpe Ratio, VaR, and correlation heatmap.
        Args:
            prices_history (pd.DataFrame): Historical prices of assets.
        """
        # Calculate portfolio returns
        weights_df = pd.DataFrame(self.broker.get_transaction_log())
        weights_df = weights_df.pivot(index='Date', columns='Ticker', values='Quantity').fillna(0)
        returns = (weights_df * prices_history.pct_change()).sum(axis=1).dropna()

        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        var = self.calculate_var(returns)

        logging.info(f"Sharpe Ratio: {sharpe_ratio}")
        logging.info(f"Value at Risk (95%): {var}")

        # Plot correlation heatmap
        self.plot_correlation_heatmap(prices_history)


