import tkinter as tk
from tkinter import simpledialog
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to get stock inputs
def get_stock_inputs():
    # Step 1: Ask the number of stocks
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    num_stocks = simpledialog.askinteger("Number of Stocks", "How many stocks do you want to enter?")
    
    if not num_stocks or num_stocks <= 0:
        print("No stocks to enter.")
        return []

    # Step 2: Create a new window to enter stock names
    root = tk.Tk()
    root.title("Enter Stock Names")
    entries = []
    stock_names = []

    def on_submit():
        # Get all the entered stock names
        nonlocal stock_names
        stock_names = [entry.get().strip().upper() for entry in entries]  # Clean inputs
        root.quit()  # Stop the mainloop
        root.destroy()  # Close the window after submission

    # Add dynamic fields for stock names
    for i in range(num_stocks):
        tk.Label(root, text=f"Stock {i+1}").pack(padx=10, pady=5)
        entry = tk.Entry(root)
        entry.pack(padx=10, pady=5)
        entries.append(entry)

    # Add a submit button
    tk.Button(root, text="Submit", command=on_submit).pack(pady=10)

    root.mainloop()  # Wait for user interaction
    return stock_names

# Function to fetch stock data
def get_stocks_data(tickers, start_date, end_date):
    """get_stocks_data retrieves historical data on prices for a list of stocks

    Args:
        tickers (list): List of stock tickers
        start_date (str): Start date in the format 'YYYY-MM-DD'
        end_date (str): End date in the format 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: A pandas dataframe with the historical data
    """
    dfs = []
    for ticker in tickers:
        try:
            # Simulate fetching data (replace this with your actual data fetching logic)
            dates = pd.date_range(start=start_date, end=end_date)
            df = pd.DataFrame({
                'Date': dates,
                'ticker': ticker,
                'Volume': np.random.randint(1000, 10000, len(dates))  # Use numpy for random volume data
            })
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Failed to fetch data for {ticker}: {e}")
    data = pd.concat(dfs) if dfs else pd.DataFrame()
    return data

# Function to rank stocks by trading volume
def rank_stocks_by_volume(df):
    """
    Ranks stocks based on their average trading volume in increasing order.

    Args:
        df (pd.DataFrame): The dataframe containing stock data with a 'Volume' column.

    Returns:
        pd.DataFrame: A dataframe with tickers and their average volumes, ranked in increasing order.
    """
    average_volumes = df.groupby('ticker')['Volume'].mean().reset_index()
    average_volumes.rename(columns={'Volume': 'average_volume'}, inplace=True)
    ranked_stocks = average_volumes.sort_values(by='average_volume', ascending=True).reset_index(drop=True)
    return ranked_stocks

# Main execution flow
if __name__ == "__main__":
    # Step 1: Get stock tickers from the user
    tickers = get_stock_inputs()

    if not tickers:
        print("No tickers entered. Exiting...")
    else:
        # Step 2: Use the entered tickers to fetch stock data
        start_date = '2022-01-01'
        end_date = '2022-12-31'

        # Fetch stock data using the entered tickers
        df = get_stocks_data(tickers, start_date, end_date)

        if df.empty:
            print("No data retrieved for the entered tickers.")
        else:
            # Step 3: Rank stocks based on their average trading volume
            ranked_df = rank_stocks_by_volume(df)

# Display in markdown format (if running in Jupyter)
            display(Markdown(f"**Ranked Stocks by Volume:**\n{ranked_df.to_markdown(index=False)}"))