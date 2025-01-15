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
        # Demander à l'utilisateur de définir le seuil si ce n'est pas passé
        if not hasattr(self, 'threshold') or self.threshold is None:
            self.threshold = float(input("Enter the stop-loss threshold (e.g., 0.1 for 10% loss): "))
        
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict, broker: 'Broker'):
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

    def get_daily_trade_count(self, date: datetime):
        """Returns the number of trades already executed on the given date."""
        daily_trades = self.transaction_log[self.transaction_log['Date'] == date]
        return len(daily_trades)

    def get_total_portfolio_value(self, market_prices: dict):
        """Calculate the total portfolio value based on the current market prices."""
        total_value = self.cash
        for ticker, position in self.positions.items():
            total_value += position.quantity * market_prices[ticker]
        return total_value

    def buy(self, ticker: str, quantity: int, price: float, date: datetime, market_prices: dict):
        """Executes a buy order for the specified ticker with additional constraints based on quantity."""
        if quantity > self.max_single_stock_trade:
            if self.verbose:
                logging.warning(
                    f"Cannot execute buy for {ticker}. Maximum allowed trade quantity is {self.max_single_stock_trade}."
                )
            return

        total_cost = price * quantity
        total_portfolio_value = self.get_total_portfolio_value(market_prices)

        # Check if daily trades max is respected
        daily_trades_for_ticker = self.transaction_log[(self.transaction_log['Date'] == date) & (self.transaction_log['Ticker'] == ticker)]
        if len(daily_trades_for_ticker) >= 1:  # check for that specific ticker
            if self.verbose:
                logging.warning(
                    f"Cannot execute buy for {ticker} on {date}. Maximum daily trades limit ({self.max_daily_trades}) reached for this ticker."
                )
            return

        # Check if the purchase respects the max exposure constraint
        current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price
        proposed_value = current_value + total_cost
        max_allowed_value = total_portfolio_value * self.max_exposure

        if proposed_value > max_allowed_value:
            if self.verbose:
                logging.warning(
                    f"Cannot buy {quantity} shares of {ticker} due to max exposure limit. "
                    f"Proposed value: {proposed_value}, Max allowed: {max_allowed_value}"
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
        """Executes a sell order for the specified ticker with additional constraints based on quantity."""
        if quantity > self.max_single_stock_trade:
            if self.verbose:
                logging.warning(
                    f"Cannot execute sell for {ticker}. Maximum allowed trade quantity is {self.max_single_stock_trade}."
                )
            return

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
    broker = Broker(cash=initial_cash, verbose=verbose)

    def __post_init__(self):
        from src.python_pro.Interactive_inputs import get_stop_loss_threshold
        self.rebalance_flag = get_rebalancing_strategy()  
        self.rebalance_flag = self.rebalance_flag() 

        self.stop_loss_threshold = get_stop_loss_threshold()
        self.broker.initialize_blockchain(self.name_blockchain)
        self.backtest_name = generate_random_name()
        

    def run_backtest(self):
        logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")
        logging.info(f"Retrieving price data for universe: {self.universe}")
        
        # Pass the custom stop-loss threshold to the StopLoss class
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

        # Initialize a DataFrame to store portfolio values
        portfolio_values = []
        
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


        logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date))}")
        df = self.broker.get_transaction_log()

        # Create the backtests folder if it does not exist
        if not os.path.exists('backtests'):
            os.makedirs('backtests')

        # Save the transaction log to CSV
        df.to_csv(f"backtests/{self.backtest_name}.csv")

        # Store the backtest results in the blockchain
        self.broker.blockchain.add_block(self.backtest_name, df.to_string())
