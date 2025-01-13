import matplotlib.pyplot as plt

class Broker:

    def get_portfolio_values_over_time(self, price_data):
        """
        Calculates the portfolio value at each time step based on historical prices.
        :param price_data: A pandas DataFrame with historical prices.
        :return: A pandas DataFrame with portfolio value over time.
        """
        portfolio_values = []
        for date in price_data.index:
            prices = price_data.loc[date].to_dict()
            portfolio_value = self.get_portfolio_value(prices)
            portfolio_values.append({
                'Date': date,
                'Portfolio Value': portfolio_value
            })
        return pd.DataFrame(portfolio_values)

# Add visualization function
def plot_backtest_results(transaction_log, portfolio_values_df):
    """
    Creates visualizations for the backtest results.
    :param transaction_log: DataFrame containing the transaction log.
    :param portfolio_values_df: DataFrame containing portfolio values over time.
    """
    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values_df['Date'], portfolio_values_df['Portfolio Value'], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot transactions
    buy_transactions = transaction_log[transaction_log['Action'] == 'BUY']
    sell_transactions = transaction_log[transaction_log['Action'] == 'SELL']

    plt.figure(figsize=(12, 6))
    plt.scatter(buy_transactions['Date'], buy_transactions['Price'], color='green', label='Buy Transactions', alpha=0.7)
    plt.scatter(sell_transactions['Date'], sell_transactions['Price'], color='red', label='Sell Transactions', alpha=0.7)
    plt.title('Transactions Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

# Update the Backtest class to include visualization after running the backtest
@dataclass
class Backtest:

    def run_backtest(self):
        logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")
        logging.info(f"Retrieving price data for universe")
        self.risk_model = self.risk_model(threshold=0.1)
        # self.initial_date to yyyy-mm-dd format
        init_ = self.initial_date.strftime('%Y-%m-%d')
        # self.final_date to yyyy-mm-dd format
        final_ = self.final_date.strftime('%Y-%m-%d')
        df = get_stocks_data(self.universe, init_, final_)

        # Initialize the DataModule
        data_module = DataModule(df)

        # Create the Information object
        info = self.information_class(s=self.s,
                                      data_module=data_module,
                                      time_column=self.time_column,
                                      company_column=self.company_column,
                                      adj_close_column=self.adj_close_column)

        # Run the backtest
        for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):
            if self.risk_model is not None:
                portfolio = info.compute_portfolio(t, info.compute_information(t))
                prices = info.get_prices(t)
                self.risk_model.trigger_stop_loss(t, portfolio, prices, self.broker)

            if self.rebalance_flag().time_to_rebalance(t):
                logging.info("-----------------------------------")
                logging.info(f"Rebalancing portfolio at {t}")
                information_set = info.compute_information(t)
                portfolio = info.compute_portfolio(t, information_set)
                prices = info.get_prices(t)
                self.broker.execute_portfolio(portfolio, prices, t)

        logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date))}")
        df = self.broker.get_transaction_log()

        # Create backtests folder if it does not exist
        if not os.path.exists('backtests'):
            os.makedirs('backtests')

        # Save to csv, use the backtest name
        df.to_csv(f"backtests/{self.backtest_name}.csv")

        # Store the backtest in the blockchain
        self.broker.blockchain.add_block(self.backtest_name, df.to_string())

        # Generate portfolio values over time
        price_data = data_module.get_price_data(self.adj_close_column)
        portfolio_values_df = self.broker.get_portfolio_values_over_time(price_data)

        # Plot the results
        plot_backtest_results(df, portfolio_values_df)
