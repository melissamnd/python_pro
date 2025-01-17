#### MODIFYIN AND ADDING FUNCTION TO DATA_MODULE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, timedelta

import os 
from pybacktestchain.data_module import get_stocks_data, DataModule, Information, FirstTwoMoments
from pybacktestchain.broker import Position, StopLoss, RebalanceFlag, Broker
from pybacktestchain.utils import generate_random_name
from typing import Dict
from numba import jit 
from python_pro.Interactive_inputs import get_rebalancing_strategy, get_stocks_data, get_stock_data

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

        self.evolution_time = []  # Liste pour stocker les dates
        self.evolution_nb_buy = []  # Liste pour stocker le nombre d'achats
        self.evolution_nb_sell = []  # Liste pour stocker le nombre de ventes

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

        # New condition: Check if the price is below a specific threshold before buying
        price_threshold = 200  # Example threshold (you can adjust this value)
        if price > price_threshold:
            if self.verbose:
                logging.warning(f"Cannot buy {ticker} on {date}. Price is above the threshold of {price_threshold}.")
            return

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

    def execute_portfolio(self, portfolio: dict, prices: dict, date: datetime):
        nb_buy = 0  # Initialize buy counter
        nb_sell = 0  # Initialize sell counter
        total_value_after_execution = self.get_portfolio_value(prices)

        # 1) Handle sell orders first to free up cash:
        for ticker, weight in portfolio.items():
            price = prices.get(ticker)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {ticker} not available on {date}")
                continue
            
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price

            # Calculate the difference and determine quantity to trade
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)

            # Execute selling
            if quantity_to_trade < 0:
                self.sell(ticker, abs(quantity_to_trade), price, date)
                nb_sell += 1  # Increment the sell counter

        # 2) Handle buy orders:
        for ticker, weight in portfolio.items():
            price = prices.get(ticker)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {ticker} not available on {date}")
                continue
        
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price

            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)
        
            # Execute buying
            if quantity_to_trade > 0:
                available_cash = self.get_cash_balance()
                cost = quantity_to_trade * price
                
                if cost <= available_cash:
                    self.buy(ticker, quantity_to_trade, price, date)
                    nb_buy += 1  # Increment the buy counter
                else:
                    # Handle insufficient cash case
                    if self.verbose:
                        logging.warning(f"Not enough cash to buy {quantity_to_trade} of {ticker} on {date}. "
                                        f"Needed: {cost}, Available: {available_cash}")
                    
                    # Buy as many shares as possible with available cash
                    quantity_to_trade = int(available_cash / price)
                    if quantity_to_trade > 0:  # Ensure we can actually buy
                        self.buy(ticker, quantity_to_trade, price, date)
                        nb_buy += 1  # Increment the buy counter

        # 3) Calculate total portfolio value after all trades:
        total_value_after_execution = self.get_portfolio_value(prices)

        # Now update the lists for the graph
        self.evolution_time.append(date)
        self.evolution_nb_buy.append(nb_buy)
        self.evolution_nb_sell.append(nb_sell)

        return total_value_after_execution, nb_sell, nb_buy



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

    def analyze(self):
        return {
            "Portfolio Total Performance": self.total_performance(),
            "Mean of the Returns": self.mean_returns(),
            "Volatility of the Returns": self.volatility_returns(),
            "Sharpe Ratio": self.sharpe_ratio(),
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
class Backtest():
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

    def run_backtest(self):

        evolution_nb_sell = []
        evolution_nb_buy = []
        evolution_portfolio_value = []
        evolution_time = []
    
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

            if prices:
                portfolio_value = self.broker.get_portfolio_value(prices)
                portfolio_values.append({'Date': t, 'PortfolioValue': portfolio_value})


                value_portfolio_after_execution, nb_sell, nb_buy = self.broker.execute_portfolio(portfolio, prices, t)
                evolution_portfolio_value.append(value_portfolio_after_execution)
                logging.info(f"Date: {t}, Portfolio: {portfolio}, Prices: {prices}")
                evolution_nb_buy.append(nb_buy)
                evolution_nb_sell.append(nb_sell)
                evolution_time.append(t)

        initial_value = self.initial_cash
        final_value = self.broker.get_portfolio_value(info.get_prices(self.final_date))
        analysis_tool = AnalysisTool(evolution_portfolio_value, initial_value, final_value)
        analysis_results = analysis_tool.analyze()
        logging.info(f"Analysis results: {analysis_results}")

        # Créer le dossier 'backtests_stats' s'il n'existe pas
        if not os.path.exists('backtests_stats'):
            os.makedirs('backtests_stats')

        # Chemin du fichier texte pour enregistrer les résultats
        file_path = f"backtests_stats/analysis_results_{self.backtest_name}.txt"

        try:
            # Ouvrir le fichier en mode écriture
            with open(file_path, 'w') as file:
                # Écrire les résultats de l'analyse dans le fichier
                for key, value in analysis_results.items():
                    file.write(f"{key}: {value}\n")
            
            logging.info(f"Analysis results saved to {file_path}")
        except Exception as e:
            logging.error(f"An error occurred while saving analysis results: {e}")

        # Continuer le reste du backtest...
        # Enregistrer les valeurs du portefeuille dans un DataFrame
        portfolio_values_df = pd.DataFrame(portfolio_values)

        # Calcul des rendements du portefeuille et ajout d'une nouvelle colonne
        portfolio_values_df['PortfolioReturn'] = portfolio_values_df['PortfolioValue'].pct_change()
        # Calcul des rendements cumulés
        portfolio_values_df['CumulativeReturn'] = (1 + portfolio_values_df['PortfolioReturn']).cumprod() - 1

        # Affichage du DataFrame
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
        #self.plot_portfolio_value_over_time(portfolio_values_df)
        #self.plot_cumulative_return_over_time(portfolio_values_df)

        """Plot portfolio value over time."""
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values_df['Date'], portfolio_values_df['PortfolioValue'], label="Portfolio Value")
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.legend()

        # Save the figure to the 'backtests_graphs' folder
        if not os.path.exists('backtests_graphs'):
            os.makedirs('backtests_graphs')

        plt.savefig(f"backtests_graphs/Portfolio_value/Portfolio_Value_Evolution_with_backtest_{self.backtest_name}.png", dpi=900)
        plt.show()

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

        fig = go.Figure()

        for stock in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[stock],
                mode='lines',
                stackgroup='one',  
                name=stock
            ))

        fig.update_layout(
            title='Portfolio Weights Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Weights',
            showlegend=True
        )

        
        folder_path = 'backtests_graphs/Portfolio_Weights'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
 
        png_path = os.path.join(folder_path, f"{self.backtest_name}_portfolio_weights.png")
        
        fig.write_image(png_path)

        fig.show()

        print(f"Portfolio weights graph saved as {png_path}")