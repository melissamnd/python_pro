#---------------------------------------------------------
# This module provides interactive functions for user input.
# It includes the following functions:
# - get date inputs from the user
# - get stock tickers from the user
# - get the selected asset allocation strategy from the user
# - get the stop-loss threshold input from the user
# - get the initial cash amount from the user
# - get the rebalance strategy from the user
#---------------------------------------------------------

import tkinter as tk
import pandas as pd
import pybacktestchain
import numpy as np
import logging
import yfinance as yf
import matplotlib.pyplot as plt

from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
from IPython.display import display, Markdown
from tkinter import simpledialog, messagebox
from pybacktestchain.data_module import get_stocks_data, get_stock_data   

class Data_module2():
    @staticmethod
    def get_stock_data(ticker, start_date, end_date):
        """Retrieve historical stock data for a single ticker using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
            data['ticker'] = ticker
            data.reset_index(inplace=True)
            return data[['Date', 'ticker', 'Adj Close', 'Volume']]
        except Exception as e:
            logging.warning(f"Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_stocks_data(tickers, start_date, end_date):
        """Retrieve historical stock data for multiple tickers."""
        dfs = []
        
        for ticker in tickers:
            df = Data_module2.get_stock_data(ticker, start_date, end_date)
            if not df.empty:
                df = df[['Date', 'Adj Close']]  
                df['Ticker'] = ticker  
                dfs.append(df)
        
        all_data = pd.concat(dfs)
        
        all_data = all_data.pivot(index='Date', columns='Ticker', values='Adj Close')
        
        all_data = all_data.dropna(how='all')
        
        return all_data


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to get stock inputs
def get_stock_inputs():
    root = tk.Tk()
    root.withdraw()  
    num_stocks = simpledialog.askinteger("Number of Stocks", "How many stocks do you want to enter?")
    
    if not num_stocks or num_stocks <= 0:
        print("No stocks to enter.")
        return []

    root = tk.Tk()
    root.title("Enter Stock Names")
    entries = []
    stock_names = []

    def on_submit():
        nonlocal stock_names
        stock_names = [entry.get().strip().upper() for entry in entries]
        root.quit()
        root.destroy()

    for i in range(num_stocks):
        tk.Label(root, text=f"Stock {i+1}").pack(padx=10, pady=5)
        entry = tk.Entry(root)
        entry.pack(padx=10, pady=5)
        entries.append(entry)

    tk.Button(root, text="Submit", command=on_submit).pack(pady=10)
    root.mainloop()
    return stock_names

# Function to get date inputs via a userform
def get_date_inputs():
    
    root = tk.Tk()
    root.withdraw()  

    
    start_date_str = simpledialog.askstring("Start Date", "Please enter the start date (YYYY-MM-DD):")
    if not start_date_str:
        return None, None 

    try:
       
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Invalid Date", "Invalid start date format. Please use YYYY-MM-DD.")
        return None, None  

    end_date_str = simpledialog.askstring("End Date", "Please enter the end date (YYYY-MM-DD):")
    if not end_date_str:
        return None, None 

    try:
        
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Invalid Date", "Invalid end date format. Please use YYYY-MM-DD.")
        return None, None 

    if start_date > end_date:
        messagebox.showerror("Invalid Date Range", "Start date must be before end date!")
        return None, None 
    
    return start_date, end_date

# Function to get the target return input from the user
def get_target_return():
    """
    Prompt the user to input a target return using a simple dialog box.
    Returns the target return as a float.
    """
    root = tk.Tk()
    root.withdraw() 
    target_return = simpledialog.askfloat(
        "Target Return", 
        "Enter the target return (e.g., 0.1 for 10%):",
        minvalue=0.0, maxvalue=1.0  
    )
    root.destroy() 
    if target_return is None:
        raise ValueError("No target return provided!")
    return target_return

# Function to get the asset allocation strategy choice from the user
def strategy_choice():
    """Select a Strategy for Asset Allocation using input box."""
    
    root = tk.Tk()
    root.withdraw() 
    
    choice = simpledialog.askstring(
        "Strategy Selection", 
        "Which strategy would you like to choose?\n"
        "1 - First Two Moments Portfolio (No Short Selling)\n"
        "2 - Long Short Portfolio (With Short Selling)\n"
        "Please enter the number of your choice (1 or 2):"
    )
    
    strategy = None
    strategy_name = None

    if choice == '1':
        from pybacktestchain.data_module import FirstTwoMoments
        strategy = FirstTwoMoments
        strategy_name = "First Two Moments Portfolio (No Short Selling)"
    elif choice == '2':
        from mm_203_python_pro.new_data_module import LongShortPortfolio
        strategy = LongShortPortfolio
        strategy_name = "Long Short Portfolio (With Short Selling)"
    else:
        simpledialog.messagebox.showwarning("Invalid input", "Please enter a valid choice (1 or 2).")
        strategy, strategy_name = strategy_choice()  # Prompt again
    
    print(f"You chose {strategy_name}")
    
    return strategy, strategy_name

# Function to get the rebalancing strategy input from the user
def get_rebalancing_strategy():

    from mm_203_python_pro.new_broker import EndOfWeek, EndOfMonth, EveryQuarter

    root = tk.Tk()
    root.withdraw() 

    choices = ["End of Week", "End of Month", "Every Quarter"]
    
    rebalancing_strategy = simpledialog.askstring("Rebalancing Strategy", 
                                                "Choose rebalancing strategy: End of Week, End of Month, or Every Quarter")
    
    if rebalancing_strategy not in choices:
        print("Invalid choice. Please choose one of the following strategies: End of Week, End of Month, or Every Quarter.")
        return None

    if rebalancing_strategy == "End of Week":
        return EndOfWeek  
    elif rebalancing_strategy == "End of Month":
        return EndOfMonth
    elif rebalancing_strategy == "Every Quarter":
        return EveryQuarter

# Function to get the stop loss input from the user
def get_stop_loss_threshold():
    def on_submit():
        nonlocal stop_loss_threshold
        try:
            user_input = threshold_entry.get()
            
            if not user_input:
                raise ValueError("Input cannot be empty.")
            
            stop_loss_threshold = float(user_input)
            
            if stop_loss_threshold <= 0:
                raise ValueError("Threshold must be a positive number.")
            
            root.quit()  
            root.destroy()  
        except ValueError as e:

            messagebox.showerror("Invalid Input", f"Invalid input: {e}")
    
    stop_loss_threshold = None
    
    root = tk.Tk()
    root.title("Enter Stop-Loss Threshold")
    
    tk.Label(root, text="Enter Stop-Loss Threshold (e.g., 0.1 for 10%):").pack(padx=10, pady=5)
    threshold_entry = tk.Entry(root)
    threshold_entry.pack(padx=10, pady=5)
    
    tk.Button(root, text="Submit", command=on_submit).pack(pady=10)
    
    root.mainloop()
    
    return stop_loss_threshold

# Function to get the initial cash input from the user
def get_initial_cash_input():
    def on_submit():
        nonlocal initial_cash
        try:
            initial_cash = float(initial_cash_entry.get())
            if initial_cash <= 0:
                raise ValueError("Initial cash must be a positive number.")
            root.quit()
            root.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid input: {e}")
    
    initial_cash = None
    
    root = tk.Tk()
    root.title("Enter Initial Cash")
    
    tk.Label(root, text="Enter Initial Cash Amount:").pack(padx=10, pady=5)
    initial_cash_entry = tk.Entry(root)
    initial_cash_entry.pack(padx=10, pady=5)
    
    tk.Button(root, text="Submit", command=on_submit).pack(pady=10)
    root.mainloop()
    
    return initial_cash