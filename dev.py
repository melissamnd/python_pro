import pandas as pd

from pybacktestchain.data_module import FirstTwoMoments, get_stocks_data
from pybacktestchain.blockchain import load_blockchain
from datetime import datetime
from mm_203_python_pro.Interactive_inputs import get_date_inputs, get_initial_cash_input, get_rebalancing_strategy, get_stop_loss_threshold, get_stock_inputs, strategy_choice
from mm_203_python_pro.new_broker import StopLoss_new, Backtest, AnalysisTool
from mm_203_python_pro.visualizing import PortfolioVisualizer, analyze_all_transactions
from mm_203_python_pro.new_data_module import LongShortPortfolio


verbose = False  

# Get initial inputs from the user
start_date, end_date = get_date_inputs()  
stop_loss_threshold = get_stop_loss_threshold()  
rebalancing_strategy = get_rebalancing_strategy()  
initial_cash = get_initial_cash_input() 
tickers = get_stock_inputs()  
strategy, strategy_name = strategy_choice()
rebalancing_strategy_instance = rebalancing_strategy()

backtest = Backtest(
    initial_date=start_date,         # Start date for the backtest chosen by the user     
    final_date=end_date,             # End date for the backtest chosen by the user
    threshold=stop_loss_threshold,   # The stop-loss threshold value chosen by the user
    information_class=strategy,          # The class used to calculate portfolio information
    risk_model=StopLoss_new,                    # The risk model used (Modified version of StopLoss)
    name_blockchain='backtest',             # The name of the blockchain where results are stored
    initial_cash=initial_cash,  # Initial amount of cash in the portfolio chosen by the user
    universe=tickers,  # The stock tickers provided by the user
    rebalance_flag=rebalancing_strategy_instance,  # The rebalancing strategy to be used chosen by the user
    verbose=verbose                      
)

# Run the backtest/graphs/stats
backtest.run_backtest()
analyze_all_transactions(backtest)

# Load the blockchain to check the results
block_chain = load_blockchain('backtest')
print(str(block_chain)) 
print(block_chain.is_valid())
