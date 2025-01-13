def get_stocks_data(tickers, start_date, end_date):
    """get_stocks_data retrieves historical data on prices for a list of stocks

    Args:
        tickers (list): List of stock tickers
        start_date (str): Start date in the format 'YYYY-MM-DD'
        end_date (str): End date in the format 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: A pandas dataframe with the historical data

    Example:
        df = get_stocks_data(['AAPL', 'MSFT'], '2000-01-01', '2020-12-31')
    """
    # get the data for each stock
    # try/except to avoid errors when a stock is not found
    dfs = []
    for ticker in tickers:
        try:
            df = get_stock_data(ticker, start_date, end_date)
            # append if not empty
            if not df.empty:
                dfs.append(df)
        except:
            logging.warning(f"Stock {ticker} not found")
    # concatenate all dataframes
    data = pd.concat(dfs)
    return data

def rank_stocks_by_volume(df):
    """
    Ranks stocks based on their average trading volume in increasing order.

    Args:
        df (pd.DataFrame): The dataframe containing stock data with a 'Volume' column.

    Returns:
        pd.DataFrame: A dataframe with tickers and their average volumes, ranked in increasing order.
    """
    # Calculate the average volume for each stock
    average_volumes = df.groupby('ticker')['Volume'].mean().reset_index()
    
    # Rename the column for clarity
    average_volumes.rename(columns={'Volume': 'average_volume'}, inplace=True)
    
    # Sort by average volume in increasing order
    ranked_stocks = average_volumes.sort_values(by='average_volume', ascending=True).reset_index(drop=True)
    
    return ranked_stocks

from pybacktestchain.data_module import get_stocks_data
from python_pro.improved_datamodule import rank_stocks_by_volume

# Get data for multiple stocks
df = get_stocks_data(['AAPL', 'MSFT', 'GOOGL'], '2022-01-01', '2022-12-31')

# Rank stocks based on their average trading volume
ranked_df = rank_stocks_by_volume(df)

print(ranked_df)