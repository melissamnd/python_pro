from pybacktestchain.broker import StopLoss
from pybacktestchain.blockchain import load_blockchain
from src.python_pro.new_broker import Backtest
from src.python_pro.Interactive_inputs import get_initial_cash_input, get_date_inputs, get_rebalancing_strategy, get_target_return

def algo_backtest():
    # Set verbosity for logging
    verbose = False  # Set to True to enable logging, or False to suppress it

    initial_cash, threshold, start_date, end_date = get_initial_parameter()
    rebalancing_strategy = strategy_choice()
    ask_user_for_comment()


    backtest = Backtest(initial_date=start_date,
            final_date=end_date,
            initial_cash=initial_cash,
            threshold = threshold,
            information_class=FirstTwoMoments,
            strategy_name=rebalancing_strategy,
            risk_model=StopLoss,
            name_blockchain='backtest',
            verbose=verbose
            )
    backtest.run_backtest()


    block_chain = load_blockchain('backtest')
    print(str(block_chain))
    # check if the blockchain is valid
    print("Blockchain valid:", block_chain.is_valid())

if __name__ == "__main__":
    algo_backtest()

