import pandas as pd 
import numpy as np

def sample_cov(prices, returns_data=False, frequency=252, log_returns=False, fix_method="spectral"):
    """
    Calculate the sample covariance matrix of returns and fix it if it's not positive semidefinite.
    """
    # Ensure input is a DataFrame
    prices = pd.DataFrame(prices) if not isinstance(prices, pd.DataFrame) else prices
    
    # Calculate returns if returns_data is False
    returns = prices if returns_data else returns_from_prices(prices, log_returns)
    
    # Compute covariance and fix non-positive semidefinite matrices
    cov_matrix = returns.cov() * frequency
    return cov_matrix if is_positive_semidefinite(cov_matrix) else fix_nonpositive_semidefinite(cov_matrix, fix_method)


def is_positive_semidefinite(matrix):
    """
    Check if a matrix is positive semidefinite using Cholesky decomposition.
    """
    try:
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


def fix_nonpositive_semidefinite(matrix, fix_method="spectral"):
    """
    Fix a non-positive semidefinite matrix (stub for now).
    """
    # You can implement specific fixes based on the method (e.g., spectral, nearest PSD matrix, etc.)
    return matrix

def returns_from_prices(prices, log_returns=False):
    if log_returns:
        returns = np.log(1 + prices.pct_change(fill_method=None)).dropna(how="all")
    else:
        returns = prices.pct_change(fill_method=None).dropna(how="all")
    return returns

def cov_to_corr(cov_matrix):
    if not isinstance(cov_matrix, pd.DataFrame):
        warnings.warn("cov_matrix is not a dataframe", RuntimeWarning)
        cov_matrix = pd.DataFrame(cov_matrix)

    Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
    corr = np.dot(Dinv, np.dot(cov_matrix, Dinv))
    return pd.DataFrame(corr, index=cov_matrix.index, columns=cov_matrix.index)