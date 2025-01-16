#### MODIFYIN AND ADDING FUNCTION TO DATA_MODULE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import os 
from pybacktestchain.data_module import get_stocks_data, DataModule, Information
from pybacktestchain.broker import Position, StopLoss, RebalanceFlag, Broker, Backtest
from pybacktestchain.utils import generate_random_name
from typing import Dict
from numba import jit 
from python_pro.Interactive_inputs import get_rebalancing_strategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datetime import timedelta, datetime

#---------------------------------------------------------
# Modifying Classes from pybacktestchain.broker
#---------------------------------------------------------

class StopLoss_new(StopLoss):
    threshold: float

    def __post_init__(self):
        #Ask user for the stop loss number
        if not hasattr(self, 'threshold') or self.threshold is None:
            self.threshold = float(input("Enter the stop-loss threshold (e.g., 0.1 for 10% loss): "))
        
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict, broker: 'Broker'):
        if not isinstance(broker.positions, dict):
            logging.error(f"Expected broker.positions to be a dictionary, but got {type(broker.positions)}")
            return  # Exit the function if it's not a dictionary

        for ticker, position in list(broker.positions.items()):
            entry_price = broker.entry_prices[ticker]
            current_price = prices.get(ticker)

            if current_price is None:
                logging.warning(f"Price for {ticker} not available on {t}")
                continue

            # Calculate the loss percentage
            loss = (current_price - entry_price) / entry_price
            if loss < -self.threshold:
                logging.info(f"Stop loss triggered for {ticker} at {t}. Selling all shares.")
                broker.sell(ticker, position.quantity, current_price, t)

class Broker_new(Broker):
# Modifying the buy and sell functions from pybacktestchain.broker to add new conditions : max daily trades and max exposure.  
# In addition, we add a function that count the number of daily trades and check if we respect the new condition.

    def __init__(self, cash, verbose=True, max_daily_trades=10, **kwargs):
        super().__init__(cash, verbose, **kwargs)
        self.max_daily_trades = max_daily_trades  
        self.positions = {}

    def get_daily_trade_count(self, date: datetime):
        """Returns the number of trades already executed on the given date."""
        daily_trades = self.transaction_log[self.transaction_log['Date'] == date]
        return len(daily_trades)

    def get_total_portfolio_value(self):
        """Calculate the total portfolio value based on the current market prices."""
        total_value = self.cash
        prices = self.get_current_prices()  # Use the adapted method to get current prices
        for ticker, position in self.positions.items():
            total_value += position.quantity * prices.get(ticker, 0)  # Use current prices
        return total_value

    def buy(self, ticker: str, quantity: int, price: float, date: datetime):
        total_cost = price * quantity

        # Check if daily trades max is respected
        daily_trades_for_ticker = self.transaction_log[(self.transaction_log['Date'] == date) & (self.transaction_log['Ticker'] == ticker)]
        if len(daily_trades_for_ticker) >= 1:  # Check for that specific ticker
            if self.verbose:
                logging.warning(
                    f"Cannot execute buy for {ticker} on {date}. Maximum daily trades limit ({self.max_daily_trades}) reached for this ticker."
                )
            return

        # Check if enough cash is available
        if self.cash < total_cost:
            if self.verbose:
                logging.warning(
                    f"Not enough cash to buy {quantity} shares of {ticker} at {price}. Available cash: {self.cash}"
                )
            return

        # Execute the buy order
        self.cash -= total_cost
        if ticker in self.positions:
            position = self.positions[ticker]
            new_quantity = position.quantity + quantity
            new_entry_price = ((position.entry_price * position.quantity) + (price * quantity)) / new_quantity
            position.quantity = new_quantity
            position.entry_price = new_entry_price
        else:
            self.positions[ticker] = Position(ticker, quantity, price)

        self.log_transaction(date, 'BUY', ticker, quantity, price)
        self.entry_prices[ticker] = price

    def sell(self, ticker: str, quantity: int, price: float, date: datetime):
        if ticker in self.positions and self.positions[ticker].quantity >= quantity:
            # Check if max daily trades limit is reached
            if self.get_daily_trade_count(date) >= self.max_daily_trades:
                if self.verbose:
                    logging.warning(
                        f"Cannot execute sell for {ticker} on {date}. Maximum daily trades limit ({self.max_daily_trades}) reached."
                    )
                return

            position = self.positions[ticker]
            position.quantity -= quantity
            self.cash += price * quantity

            if position.quantity == 0:
                del self.positions[ticker]
                del self.entry_prices[ticker]
            self.log_transaction(date, 'SELL', ticker, quantity, price)
        else:
            if self.verbose:
                logging.warning(
                    f"Not enough shares to sell {quantity} shares of {ticker}. Position size: {self.positions.get(ticker, 0)}"
                )


#---------------------------------------------------------
# Creating new classes for portfolio analysis
#---------------------------------------------------------


#Creation of a new class that computes different statistics to analyse the portfolio. The class includes the below functions:
#   Computation of the performance of the portfolio
#   Calculation or returns, mean, vol, Sharpe Ratio and VaR


@dataclass
class AnalysisTool:
    def __init__(self, portfolio_values, initial_value, final_value, risk_free_rate=0.03):
        self.portfolio_values = np.array(portfolio_values)
        self.initial_value = initial_value
        self.final_value = final_value
        self.risk_free_rate = risk_free_rate

    def total_performance(self):
        return (self.final_value - self.initial_value) / self.initial_value

    def calculate_returns(self):
        return np.diff(self.portfolio_values) / self.portfolio_values[:-1]
    
    def mean_returns(self):
        returns = self.calculate_returns()
        return np.mean(returns)
    
    def volatility_returns(self):
        returns = self.calculate_returns()
        return np.std(returns)
        
    def sharpe_ratio(self):
        returns = self.calculate_returns()
        excess_returns = returns - self.risk_free_rate
        return np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0

    def calculate_var(self, confidence_level=0.95):
        returns = self.calculate_returns()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var

    def analyze(self):
        return {
            "Portfolio Total Performance": self.total_performance(),
            "Mean of the Returns": self.mean_returns(),
            "Volatility of the Returns": self.volatility_returns(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "VaR (95% Confidence)": self.calculate_var(confidence_level=0.95)  # VaR at 95% confidence level
        }
#---------------------------------------------------------
# Creating new classes allowing for new rebalances
#---------------------------------------------------------

# Creation of new classes to allow for more frequent rebalances : every week/month/quarter

@dataclass
class RebalanceFlag:
    def time_to_rebalance(self, t: datetime):
        pass


class EndOfWeek:
    def time_to_rebalance(self, t):
        """Rebalances at the end of the week (Friday)"""
        pd_date = pd.Timestamp(t)
        return pd_date.weekday() == 4  # 4 is Friday

class EndOfMonth:
    def time_to_rebalance(self, t):
        """Rebalances at the end of the month"""
        pd_date = pd.Timestamp(t)
        last_business_day = pd_date + pd.offsets.BMonthEnd(0)
        return pd_date == last_business_day

class EveryQuarter:
    def time_to_rebalance(self, t):
        """Rebalances at the start of each quarter"""
        pd_date = pd.Timestamp(t)
        return pd_date.month in [1, 4, 7, 10] and pd_date.day == 1


#---------------------------------------------------------
# Made changes to backtest function to account for modifications
#---------------------------------------------------------

# Allow new rebalances
# Allow dynamic threshold
# Allow for dynamic universe
# Plot graphs

@dataclass
class Backtest2:
    initial_date: datetime
    final_date: datetime
    initial_cash: int = 1000000  # Default initial cash
    threshold: float = 0.1  
    universe: list = None  # list of stock tickers
    information_class: type = Information
    s: timedelta = timedelta(days=360)
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column: str = 'Adj Close'
    rebalance_flag: type = EndOfMonth # Default rebalancing is monthly
    risk_model: type = StopLoss
    verbose: bool = True
    name_blockchain: str = 'backtest'
    broker = Broker_new(cash=initial_cash, verbose=verbose)

    def __post_init__(self):
        from src.python_pro.Interactive_inputs import get_stop_loss_threshold
        self.rebalance_flag = get_rebalancing_strategy()  
        self.rebalance_flag = self.rebalance_flag() 

        self.stop_loss_threshold = get_stop_loss_threshold()
        self.broker.initialize_blockchain(self.name_blockchain)
        self.backtest_name = generate_random_name()

        # Initialize weights and portfolio_values
        self.weights = {}
        self.portfolio_values = []

    def run_backtest(self):
        from python_pro.new_broker import Broker_new
        logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")
        logging.info(f"Retrieving price data for universe: {self.universe}")
        
        self.risk_model = self.risk_model(self.stop_loss_threshold)
        
        # Convert dates to strings
        init_ = self.initial_date.strftime('%Y-%m-%d')
        final_ = self.final_date.strftime('%Y-%m-%d')
        
        # Retrieve stock data
        df = get_stocks_data(self.universe, init_, final_)

        # Initialize the DataModule
        data_module = DataModule(df)

        # Create the Information object
        info = self.information_class(s=self.s,
                                    data_module=data_module,
                                    time_column=self.time_column,
                                    company_column=self.company_column,
                                    adj_close_column=self.adj_close_column)

        # Run the backtest
        for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):
            if self.risk_model is not None:
                portfolio = info.compute_portfolio(t, info.compute_information(t))
                prices = info.get_prices(t)
                self.risk_model.trigger_stop_loss(t, portfolio, prices, self.broker)
        
            if self.rebalance_flag.time_to_rebalance(t):
                logging.info("-----------------------------------")
                logging.info(f"Rebalancing portfolio at {t}")
                information_set = info.compute_information(t)
                portfolio = info.compute_portfolio(t, information_set)
                prices = info.get_prices(t)
                self.broker.execute_portfolio(portfolio, prices, t)

            if prices:
                portfolio_value = self.broker.get_portfolio_value(prices)
                self.portfolio_values.append({'Date': t, 'PortfolioValue': portfolio_value})

        # Save the portfolio values to a DataFrame
        portfolio_values_df = pd.DataFrame(self.portfolio_values)

        # Calculate portfolio returns and add them as a new column
        portfolio_values_df['PortfolioReturn'] = portfolio_values_df['PortfolioValue'].pct_change()
        # Calculate cumulative returns
        portfolio_values_df['CumulativeReturn'] = (1 + portfolio_values_df['PortfolioReturn']).cumprod() - 1

        # Print the DataFrame directly
        print(portfolio_values_df)

        logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date))}")
        df = self.broker.get_transaction_log()

        # Create the backtests folder if it does not exist
        if not os.path.exists('backtests'):
            os.makedirs('backtests')

        # Save the transaction log to CSV
        df.to_csv(f"backtests/{self.backtest_name}.csv")

        # Store the backtest results in the blockchain
        self.broker.blockchain.add_block(self.backtest_name, df.to_string())

        # Now plot and save the graphs
        self.plot_portfolio_value_over_time()
        self.plot_weights(self.weights)
        self.plot_historical_prices(df)
        self.plot_portfolio_allocation(self.broker.positions)

    def plot_portfolio_weights(self, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        portfolio_weights = []
        stock_list = None

        info = self.information_class(
            s=self.s,
            data_module=DataModule(get_stocks_data(self.universe, '2015-01-01', '2023-01-01')),
            time_column=self.time_column,
            company_column=self.company_column,
            adj_close_column=self.adj_close_column
        )

        for date in dates:
            information_set = info.compute_information(date)
            portfolio = info.compute_portfolio(date, information_set)
            portfolio_weights.append(portfolio)
            if stock_list is None:
                stock_list = list(portfolio.keys())

        df = pd.DataFrame(portfolio_weights, index=dates, columns=stock_list).fillna(0)

        # Create a Plotly figure
        fig = go.Figure()

        # Add a trace for each stock in the portfolio
        for stock in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[stock],
                mode='lines',
                stackgroup='one',  # This creates the stacked effect
                name=stock
            ))

        fig.update_layout(
            title='Portfolio Weights Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Weights',
            showlegend=True
        )

    def plot_portfolio_value_over_time(self):
        """Plot portfolio value over time."""
        plt.figure(figsize=(12, 6))
        plt.plot([p['Date'] for p in self.portfolio_values], [p['PortfolioValue'] for p in self.portfolio_values], label="Portfolio Value")
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()

        # Save the figure to the 'backtests_graphs' folder
        if not os.path.exists('backtests_graphs'):
            os.makedirs('backtests_graphs')

        plt.savefig(f"backtests_graphs/Portfolio_Value_Evolution_with_backtest_{self.backtest_name}.png", dpi=900)
        plt.show()

    def plot_weights(self, weights):
        """Plot portfolio weights as a horizontal bar chart."""
        if not weights:
            print("No weights available for plotting.")
            return

        tickers = list(weights.keys())
        weight_values = list(weights.values())
        plt.figure(figsize=(12, 6))
        plt.barh(tickers, weight_values, color='skyblue')
        plt.title("Portfolio Weights")
        plt.xlabel("Weight (%)")
        plt.ylabel("Ticker")
        plt.show()

        # Save the figure to the 'backtests_graphs' folder
        if not os.path.exists('backtests_graphs'):
            os.makedirs('backtests_graphs')

        plt.savefig(f"backtests_graphs/Portfolio_Weights_{self.backtest_name}.png", dpi=900)
        plt.show()

    def plot_historical_prices(self, df):
        """Plots historical prices from a DataFrame."""
        tickers = df.columns 
        plt.figure(figsize=(12, 6))

        for ticker in tickers:
            plt.plot(df.index, df[ticker], label=ticker)

        plt.title("Historical Prices")
        plt.xlabel("Date")
        plt.ylabel("Adjusted Close Price")
        plt.legend()
        plt.show()

        # Save the figure to the 'backtests_graphs' folder
        plt.savefig(f"backtests_graphs/Historical_Prices_{self.backtest_name}.png", dpi=900)
        plt.show()

    def plot_portfolio_allocation(self, portfolio):
        """Plot the portfolio allocation as a pie chart."""
        labels = list(portfolio.keys())
        sizes = [position.quantity for position in portfolio.values()]
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title("Portfolio Allocation")
        plt.axis('equal')  
        plt.show()

        # Save the figure to the 'backtests_graphs' folder
        plt.savefig(f"backtests_graphs/Portfolio_Allocation_{self.backtest_name}.png", dpi=900)
        plt.show()
