
#### MY DEV.PY FILE
import pandas as pd
from pybacktestchain.data_module import FirstTwoMoments, get_stocks_data
#from pybacktestchain.broker import StopLoss -- we will use our own StopLoss function
from pybacktestchain.blockchain import load_blockchain
from datetime import datetime
from src.python_pro.Interactive_inputs import get_date_inputs, get_initial_cash_input, get_rebalancing_strategy, get_stop_loss_threshold, get_stock_inputs, strategy_choice
from src.python_pro.new_broker import StopLoss_new, Backtest, AnalysisTool
from src.python_pro.visualizing import PortfolioVisualizer, analyze_all_transactions
from src.python_pro.new_data_module import LongShortPortfolio

# Set verbosity for logging
verbose = False  # Set to True to enable logging, or False to suppress it

# Get initial inputs from the user
start_date, end_date = get_date_inputs()  # Get start and end date from the user input
stop_loss_threshold = get_stop_loss_threshold()  # Get the stop-loss threshold
rebalancing_strategy = get_rebalancing_strategy()  # Get the rebalancing strategy
initial_cash = get_initial_cash_input()  # Get the initial cash amount
tickers = get_stock_inputs()  # Get the list of stock tickers
strategy, strategy_name = strategy_choice()
rebalancing_strategy_instance = rebalancing_strategy()

# Now we need to use the Backtest class, correctly passing the parameters.
backtest = Backtest(
    initial_date=start_date,         # Start date for the backtest (DYNAMIC)      
    final_date=end_date,             # End date for the backtest (DYNAMIC)
    threshold=stop_loss_threshold,   # The stop-loss threshold value (DYNAMIC)
    information_class=strategy,          # The class used to calculate portfolio information
    risk_model=StopLoss_new,                    # The risk model used (e.g., Modified version of StopLoss)
    name_blockchain='backtest',             # The name of the blockchain where results are stored
    initial_cash=initial_cash,  # Initial amount of cash in the portfolio (DYNAMIC)
    universe=tickers,  # The stock tickers provided by the user
    rebalance_flag=rebalancing_strategy_instance,  # The rebalancing strategy to be used
    verbose=verbose                        # Whether to show logs or not
)

# Run the backtest
backtest.run_backtest()
analyze_all_transactions(backtest)

# Load the blockchain to check the results
block_chain = load_blockchain('backtest')
print(str(block_chain)) # Print the blockchain content (backtest results)
# Check if the blockchain is valid
print(block_chain.is_valid())
