#---------------------------------------------------------
# This module provides functions for analyzing and visualizing the results of a backtest.
# It includes the following functions:
# - plot_historical_prices: Plots the historical adjusted closing prices of the selected stocks.
# - plot_var: Plots the Value at Risk (VaR) over time for the portfolio.
# - calculate_returns: Calculates the portfolio's returns based on its historical value.
# - calculate_var: Calculates the Value at Risk (VaR) based on the portfolio's returns.
# - analyze_all_transactions: Analyzes all buy and sell transactions for each ticker, calculating 
#   the total quantity bought/sold, average buy/sell prices, and visualizes the buy/sell distribution.
#---------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logging
import os

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

    def plot_historical_prices(self, df):
        """Plots historical prices from a DataFrame."""
        
        if df.empty:
            logging.warning("No data available to plot historical prices.")
            return
        
        tickers = df['ticker'].unique()  
        plt.figure(figsize=(12, 6))

        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker]
            
            if not ticker_data.empty: 
                plt.plot(ticker_data['Date'], ticker_data['Adj Close'], label=ticker)
            else:
                logging.warning(f"No data available for ticker {ticker}.")

        plt.title("Historical Prices of Selected Stocks")
        plt.xlabel("Date")
        plt.ylabel("Adjusted Close Price ($)")
        plt.legend()

        plt.show()

        if not os.path.exists('backtests_graphs'):
            os.makedirs('backtests_graphs')

        plt.savefig(f"backtests_graphs/Historical_Prices_{self.data['backtest_name']}.png", dpi=900)
        plt.show()

    def plot_var(self):
        """Plot Value at Risk (VaR) over time"""
        returns = self.calculate_returns()
        var = [self.calculate_var(confidence_level=0.95)] * len(returns)  

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(returns)), var, label="VaR (95% confidence)", color="red")
        plt.title("Value at Risk (VaR) Over Time")
        plt.xlabel("Time")
        plt.ylabel("VaR")
        plt.legend()

        if not os.path.exists('backtests_graphs'):
            os.makedirs('backtests_graphs')
        
        plt.savefig(f"backtests_graphs/Value_at_Risk_{self.data['initial_value']}_{self.data['final_value']}.png", dpi=900)
        plt.show()

    def calculate_returns(self):
        """Calculates the returns of the portfolio"""
        return np.diff(self.data['PortfolioValue']) / self.data['PortfolioValue'][:-1]

    def calculate_var(self, confidence_level=0.95):
        """Calculates the Value at Risk (VaR)"""
        returns = self.calculate_returns()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var

    
def analyze_all_transactions(backtest_instance):
    """
    Analyse all the transaction for every ticker chosen by the user.
    """

    backtest_name = backtest_instance.backtest_name  

    transaction_log_path = f'backtests/{backtest_name}.csv'
        
    if os.path.exists(transaction_log_path):
        
        df = pd.read_csv(transaction_log_path)
            
        if not os.path.exists('backtest_stats'):
            os.makedirs('backtest_stats')

        stats_list = []  
            
        tickers = df['Ticker'].unique()
        print(f"Unique tickers found: {tickers}")

        for ticker in tickers:
            print(f"\nAnalyzing transactions for {ticker}:", flush=True)
                
            ticker_df = df[df['Ticker'] == ticker]
                
            buy_ticker = ticker_df[ticker_df['Action'] == 'BUY']
            sell_ticker = ticker_df[ticker_df['Action'] == 'SELL']

            total_bought = buy_ticker['Quantity'].sum()
            total_sold = sell_ticker['Quantity'].sum()

            avg_buy_price = (buy_ticker['Quantity'] * buy_ticker['Price']).sum() / total_bought if total_bought > 0 else 0
            avg_sell_price = (sell_ticker['Quantity'] * sell_ticker['Price']).sum() / total_sold if total_sold > 0 else 0

            print(f"Total {ticker} bought: {total_bought} shares", flush=True)
            print(f"Total {ticker} sold: {total_sold} shares", flush=True)
            print(f"Average buy price for {ticker}: ${avg_buy_price:.2f}", flush=True)
            print(f"Average sell price for {ticker}: ${avg_sell_price:.2f}", flush=True)

            stats_list.append({
                "Ticker": ticker,
                "Total Bought": total_bought,
                "Total Sold": total_sold,
                "Average Buy Price": avg_buy_price,
                "Average Sell Price": avg_sell_price
            })


            plt.figure(figsize=(6, 6))
            plt.pie([total_bought, total_sold], labels=["Bought", "Sold"], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
            plt.title(f"Buy and Sell Distribution for {ticker}")
            plt.axis('equal') 
            plt.savefig(f"backtests_graphs/Buy_and_Sell/{ticker}_Buy_Sell_Distribution.png")
            plt.show()


        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv('backtest_transac_stats/transaction_analysis.csv', index=False)
        print(stats_df)
        print(f"Transaction analysis saved to 'backtest_stats/transaction_analysis_.csv'", flush=True)

    else:
        print(f"Le fichier de backtest {transaction_log_path} n'existe pas.", flush=True)
        return None