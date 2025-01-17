import numpy as np
import logging
import yfinance as yf
import pandas as pd 
import logging 
import numpy as np

from scipy.optimize import minimize
from sec_cik_mapper import StockMapper
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.optimize import minimize
from pybacktestchain.data_module import Information

# Volatility minimization with return objective
@dataclass
class LongShortPortfolio(Information):
    def compute_portfolio(self, t: datetime, information_set):
        try:
            mu = information_set['expected_return']
            Sigma = information_set['covariance_matrix']
            n = len(mu)
            
            # Objective function: Minimize portfolio variance
            obj = lambda x: x.dot(Sigma).dot(x)
            
            # Constraints: The sum of weights must equal 1 (fully invested portfolio)
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Allow short selling, bounds for each asset can be -1 (short) to +1 (long)
            bounds = [(-1.0, 1.0)] * n
            
            # Initial guess: Equal weights
            x0 = np.ones(n) / n
            
            # Minimize objective function
            res = minimize(obj, x0, constraints=cons, bounds=bounds)

            # Prepare portfolio dictionary
            portfolio = {k: None for k in information_set['companies']}

            if res.success:
                for i, company in enumerate(information_set['companies']):
                    portfolio[company] = res.x[i]
            else:
                raise Exception("Optimization did not converge")

            return portfolio
        except Exception as e:
            # Return equal weight portfolio if error occurs
            logging.warning("Error computing portfolio, returning equal weight portfolio")
            logging.warning(e)
            return {k: 1/len(information_set['companies']) for k in information_set['companies']}

    def compute_information(self, t: datetime):
        # Get the data module
        data = self.slice_data(t)
        information_set = {}

        # Sort data by ticker and date
        data = data.sort_values(by=[self.company_column, self.time_column])

        # Compute returns per company
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()

        # Expected return by company
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # Covariance matrix
        data = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
        data = data.dropna(axis=0)
        covariance_matrix = data.cov()
        covariance_matrix = covariance_matrix.to_numpy()

        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data.columns.to_numpy()
        return information_set
