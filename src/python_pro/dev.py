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

