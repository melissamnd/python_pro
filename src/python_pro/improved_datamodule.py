#%% Import necessary libraries
import tkinter as tk
from tkinter import simpledialog, ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis
from IPython.display import display, Markdown
import yfinance as yf
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to get stock inputs
def get_stock_inputs():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
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
    def on_submit():
        nonlocal start_date, end_date
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()

        try:
            # Validate the dates
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
            if start_date > end_date:
                raise ValueError("Start date must be before end date!")
            root.quit()
            root.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid date format or range: {e}")
    
    start_date = None
    end_date = None
    
    root = tk.Tk()
    root.title("Select Date Range")
    
    tk.Label(root, text="Start Date (YYYY-MM-DD):").pack(padx=10, pady=5)
    start_date_entry = tk.Entry(root)
    start_date_entry.pack(padx=10, pady=5)
    
    tk.Label(root, text="End Date (YYYY-MM-DD):").pack(padx=10, pady=5)
    end_date_entry = tk.Entry(root)
    end_date_entry.pack(padx=10, pady=5)
    
    tk.Button(root, text="Submit", command=on_submit).pack(pady=10)
    root.mainloop()
    
    return start_date, end_date

# Function to fetch stock data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    """Retrieve historical stock data for a single ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
        data['ticker'] = ticker
        data.reset_index(inplace=True)
        return data[['Date', 'ticker', 'Close', 'Volume']]
    except Exception as e:
        logging.warning(f"Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()

# Function to fetch data for multiple stocks
def get_stocks_data(tickers, start_date, end_date):
    """Retrieve historical stock data for multiple tickers."""
    dfs = []
    for ticker in tickers:
        df = get_stock_data(ticker, start_date, end_date)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs) if dfs else pd.DataFrame()

# Function to rank stocks by trading volume
def rank_stocks_by_volume(df):
    """Ranks stocks based on their average trading volume."""
    average_volumes = df.groupby('ticker')['Volume'].mean().reset_index()
    average_volumes.rename(columns={'Volume': 'average_volume'}, inplace=True)
    ranked_stocks = average_volumes.sort_values(by='average_volume', ascending=True).reset_index(drop=True)
    return ranked_stocks

@dataclass
class DataModule:
    data: pd.DataFrame

@dataclass
class Information:
    s: timedelta = timedelta(days=360)  # Time step (rolling window)
    data_module: DataModule = None
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column: str = 'Close'

    def slice_data(self, t: datetime):
        """Slices data within the rolling window [t-s, t)."""
        data = self.data_module.data
        s = self.s
        t = pd.Timestamp(t).tz_localize(None)
        data[self.time_column] = pd.to_datetime(data[self.time_column]).dt.tz_localize(None)
        return data[(data[self.time_column] >= t - s) & (data[self.time_column] < t)]

    def get_prices(self, t: datetime):
        """Gets the latest prices at time t."""
        data = self.slice_data(t)
        prices = data.groupby(self.company_column)[self.adj_close_column].last().to_dict()
        return pd.DataFrame(list(prices.items()), columns=['Ticker', 'Price'])

    def compute_skewness_and_kurtosis(self, t: datetime):
        """Computes skewness and kurtosis for each stock."""
        data = self.slice_data(t)
        grouped = data.groupby(self.company_column)[self.adj_close_column]
        skewness_df = grouped.apply(lambda x: skew(x, nan_policy='omit')).reset_index(name='Skewness')
        kurtosis_df = grouped.apply(lambda x: kurtosis(x, nan_policy='omit')).reset_index(name='Kurtosis')
        return skewness_df, kurtosis_df

#Adding graph 

def plot_historical_prices(df):
    """
    Plots historical prices for each ticker.

    Args:
        df (pd.DataFrame): DataFrame containing historical price data with columns ['Date', 'ticker', 'Close'].
    """
    tickers = df['ticker'].unique()
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker]
        plt.plot(ticker_data['Date'], ticker_data['Close'], label=ticker)
    
    plt.title("Historical Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main execution flow
if __name__ == "__main__":
    # Step 1: Get stock tickers from the user
    tickers = get_stock_inputs()
    if not tickers:
        print("No tickers entered. Exiting...")
    else:
        # Step 2: Get date range from the user
        start_date, end_date = get_date_inputs()
        if not start_date or not end_date:
            print("No dates provided. Exiting...")
        else:
            # Step 3: Fetch historical stock data
            df = get_stocks_data(tickers, start_date, end_date)

            if df.empty:
                print("No data retrieved for the entered tickers.")
            else:
                # Step 4: Plot historical prices
                print("Plotting Historical Prices.")
                plot_historical_prices(df)

                # Step 5: Rank stocks by average trading volume
                ranked_df = rank_stocks_by_volume(df)
                display(Markdown("### Ranked Stocks by Volume:"))
                display(ranked_df)

                # Step 6: Initialize DataModule and Information
                data_module = DataModule(data=df)
                information = Information(data_module=data_module)

                # Step 7: Specify a time point for analysis
                t = datetime.strptime(end_date, '%Y-%m-%d')

                # Step 8: Get latest prices
                prices = information.get_prices(t)
                display(Markdown("### Latest Prices:"))
                display(prices)

                # Step 9: Compute skewness and kurtosis
                skewness_df, kurtosis_df = information.compute_skewness_and_kurtosis(t)
                display(Markdown("### Skewness:"))
                display(skewness_df)
                display(Markdown("### Kurtosis:"))
                display(kurtosis_df)

