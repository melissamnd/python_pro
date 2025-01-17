# Python_pro

This project is a portfolio backtesting tool built on the `pybacktestchain` library, extended with custom functionality for better portfolio analysis, risk management, and dynamic trading strategies.

## Key Features

- **Custom StopLoss Class**: User-defined stop-loss threshold that triggers a sell order when a position’s loss exceeds the specified limit.
- **Custom Broker Class**: Added conditions for managing daily trades and exposure:
  - **Max Daily Trades**: Limits the number of trades executed for a specific ticker each day.
  - **Exposure Control**: Ensures that trades respect maximum portfolio exposure.
- **Portfolio Analysis Tools**: Calculations for portfolio performance, risk metrics, and Sharpe ratio.
- **Dynamic Rebalancing Strategies**: Supports weekly, monthly, and quarterly rebalancing strategies.
- **Graphing Tools**: Enhanced visualizations for portfolio value, returns, VaR (Value at Risk), and more.
- **Optimized for Asset Allocation**: Implements strategies such as Long-Short portfolio for dynamic asset management.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/portfolio-backtest-tool.git
   cd portfolio-backtest-tool

2. **Install dependencies:**
Make sure you have Python 3.6 or higher installed. You can then install the required dependencies using the following command:

    pip install -r requirements.txt

This will install all the necessary libraries for the project, including dependencies for portfolio backtesting, data handling, and visualizations.

## Usage

### 1. Configure Backtest Parameters:
When running the backtest, you will need to provide some parameters. These can be set interactively by the program:

- **Start Date and End Date**: The time period for which you want to run the backtest. These will be asked during execution.
- **Stop-Loss Threshold**: A value indicating at which percentage loss a position should be automatically sold to limit further losses.
- **Rebalancing Strategy**: You can choose the frequency for portfolio rebalancing (weekly, monthly, or quarterly).
- **Initial Cash**: The starting cash amount for the backtest. This will be provided interactively.
- **Stock Tickers**: The stock symbols (tickers) for the assets to include in your portfolio for backtesting. You will input these interactively.

### 2. Run the Backtest:
Once all the parameters are set, you can run the backtest using the following Python script:

```python
from src.python_pro.new_broker import StopLoss_new, Backtest
from src.python_pro.Interactive_inputs import get_date_inputs, get_initial_cash_input, get_rebalancing_strategy, get_stop_loss_threshold, get_stock_inputs, strategy_choice

# Get initial inputs from the user
start_date, end_date = get_date_inputs()  
stop_loss_threshold = get_stop_loss_threshold()  
rebalancing_strategy = get_rebalancing_strategy()  
initial_cash = get_initial_cash_input() 
tickers = get_stock_inputs()  
strategy, strategy_name = strategy_choice()
rebalancing_strategy_instance = rebalancing_strategy()

# Create a backtest instance
backtest = Backtest(
    initial_date=start_date,         
    final_date=end_date,             
    threshold=stop_loss_threshold,   
    information_class=strategy,      
    risk_model=StopLoss_new,         
    name_blockchain='backtest',      
    initial_cash=initial_cash,      
    universe=tickers,               
    rebalance_flag=rebalancing_strategy_instance,  
    verbose=False                   
)

# Run the backtest and visualize results
backtest.run_backtest()

# Analyze transactions
from src.python_pro.visualizing import analyze_all_transactions
analyze_all_transactions(backtest)

# Check blockchain for results
from pybacktestchain.blockchain import load_blockchain
block_chain = load_blockchain('backtest')
print(str(block_chain)) 
print(block_chain.is_valid())
```

### 3. Visualize Results:
Once the backtest has completed, graphs and analysis will be generated. This includes:

- Portfolio Value Over Time
- Cumulative Return Over Time
- Distribution of Portfolio Returns
- Sharpe Ratio Over Time
- Maximum Drawdown Over Time

Graphs will be saved automatically in the `backtests_graphs` directory, and detailed results will be logged.

### 4. Access Transaction Details:
After running the backtest, transaction details (such as buy/sell distribution) will be available in a CSV file under `backtest_stats/transaction_analysis.csv`. You can analyze all transactions using the `analyze_all_transactions` function, which breaks down each ticker's activity, including the number of shares bought and sold and average prices.

### 5. Blockchain Verification:
The backtest results are stored in the blockchain. You can check the validity of the blockchain and load the backtest data with the following code:

```python
from pybacktestchain.blockchain import load_blockchain

block_chain = load_blockchain('backtest')
print(str(block_chain)) 
print(block_chain.is_valid())
```

This will display the blockchain content and verify whether the results are valid.

### 6. Running Graphs:
After the backtest, various graphs are generated for deeper insights into the portfolio performance. These include:

- Portfolio Value Over Time
- Cumulative Return
- Distribution of Portfolio Returns
- Sharpe Ratio
- Maximum Drawdown

You can find all of the generated plots in the `backtests_graphs` folder. They will be saved as PNG files, and each graph is labeled with its respective name. Here's a preview of the saved directories:

- `backtests_graphs/Portfolio_value/`
- `backtests_graphs/Cumulative_return/`
- `backtests_graphs/Returns_Distribution/`
- `backtests_graphs/Sharpe_Ratio/`
- `backtests_graphs/Max_Drawdown/`

### Example of Output:
After running the backtest and analyzing transactions, you will see outputs in the terminal, such as:

- Portfolio performance metrics
- Number of buys/sells and average buy/sell prices for each stock
- Graphs that track portfolio performance over time

### Additional Information

#### Customization

- **Risk Model**: You can modify the risk model by adjusting the threshold in the `StopLoss_new` class, which will trigger stop-loss actions based on the percentage loss you define.

- **Rebalancing Strategy**: The system currently supports weekly, monthly, and quarterly rebalancing strategies. You can choose your preferred strategy when initializing the backtest.

- **Max Daily Trades & Max Exposure**: The broker now supports constraints for limiting the number of trades per day (`max_daily_trades`) and exposure to a single asset.

#### Future Enhancements

- More advanced portfolio optimization strategies (e.g., incorporating other risk models like Mean-Variance Optimization)
- Integration with external APIs for real-time data and portfolio execution
- Additional risk management models such as Value-at-Risk (VaR), Drawdown limits, etc.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`python_pro` was created by Melissa Mesnard. It is licensed under the terms of the MIT license.

## Credits

`python_pro` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
