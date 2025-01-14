#%% Import necessary libraries
import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
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
class FirstTwoMoments:
    s: timedelta = timedelta(days=360)
    data_module: DataModule = None
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column: str = 'Close'

    def slice_data(self, t: datetime):
        data = self.data_module.data
        s = self.s
        t = pd.Timestamp(t).tz_localize(None)
        data[self.time_column] = pd.to_datetime(data[self.time_column]).dt.tz_localize(None)
        return data[(data[self.time_column] >= t - s) & (data[self.time_column] < t)]

    def compute_information(self, t: datetime):
        data = self.slice_data(t)
        information_set = {}
        data = data.sort_values(by=[self.company_column, self.time_column])
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()
        data = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column).dropna(axis=0)
        information_set['covariance_matrix'] = data.cov().to_numpy()
        information_set['companies'] = data.columns.to_numpy()
        return information_set

    def compute_portfolio(self, t: datetime, information_set):
        try:
            mu = information_set['expected_return']
            Sigma = information_set['covariance_matrix']
            gamma = 1
            n = len(mu)
            obj = lambda x: -x.dot(mu) + gamma / 2 * x.dot(Sigma).dot(x)
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(0.0, 1.0)] * n
            x0 = np.ones(n) / n
            res = minimize(obj, x0, constraints=cons, bounds=bounds)
            if res.success:
                return dict(zip(information_set['companies'], res.x))
            else:
                raise ValueError("Optimization did not converge.")
        except Exception as e:
            logging.warning(f"Optimization failed: {e}")
            return {k: 1/len(information_set['companies']) for k in information_set['companies']}

    

# Function to plot historical prices
def plot_historical_prices(df):
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

                # Function to plot portfolio allocation as a pie chart
def plot_portfolio_allocation(portfolio):
    labels = list(portfolio.keys())
    sizes = list(portfolio.values())  
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Portfolio Allocation")
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
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
            # Step 3: Load historical data for the tickers
            df = get_stocks_data(tickers, start_date, end_date)
            if df.empty:
                print("No data retrieved for the entered tickers.")
            else:
                print("Data loaded successfully.")
                
                # Step 4: Plot historical prices
                plot_historical_prices(df)

                # Step 5: Rank stocks by trading volume
                ranked_df = rank_stocks_by_volume(df)
                display(Markdown("### Ranked Stocks by Volume:"))
                display(ranked_df)

                # Step 6: Create a DataModule object
                data_module = DataModule(data=df)

                # Step 7: Initialize the FirstTwoMoments class
                first_two_moments = FirstTwoMoments(data_module=data_module)

                # Step 8: Specify a date for the portfolio analysis (end of the date range)
                t = datetime.strptime(end_date, '%Y-%m-%d')

                # Step 9: Compute the information set
                information_set = first_two_moments.compute_information(t)
                display(Markdown("### Information Set:"))
                print(f"Expected Returns: {information_set['expected_return']}")
                print(f"Covariance Matrix:\n{information_set['covariance_matrix']}")

                # Step 10: Compute the optimized portfolio
                portfolio = first_two_moments.compute_portfolio(t, information_set)

                # Step 11: Display the portfolio weights
                display(Markdown("### Optimized Portfolio:"))
                display(pd.DataFrame(list(portfolio.items()), columns=['Ticker', 'Weight']))

                # Step 12: Plot portfolio allocation
                plot_portfolio_allocation(portfolio)


