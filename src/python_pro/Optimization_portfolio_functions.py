import numpy as np  
from scipy import optimize as opt  

# Volatility minimization with return objective
def port_minvol_ro(expected_return, covariance_matrix, ro):
    """
    Minimize portfolio volatility for a target return.
    """
    def objective(W, R, C, ro):
        # Portfolio variance
        varp = np.dot(np.dot(W.T, C), W)
        return varp**0.5  # Minimize volatility
    
    n = len(covariance_matrix)
    W = np.ones([n]) / n  # Initial weights: equally distributed
    bounds = [(0., 1.) for _ in range(n)]  # No short selling
    constraints = [
        {'type': 'eq', 'fun': lambda W: np.sum(W) - 1.},  # Weights sum to 1
        {'type': 'eq', 'fun': lambda W: np.dot(W.T, expected_return) - ro}  # Target return
    ]
    
    optimized = opt.minimize(
        objective, W, (expected_return, covariance_matrix, ro),
        method='SLSQP', constraints=constraints, bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-08}
    )
    return optimized.x

# Volatility minimization
def port_minvol(expected_return, covariance_matrix):
    """
    Minimize portfolio volatility.
    """
    def objective(W, R, C):
        # Portfolio variance
        varp = np.dot(np.dot(W.T, C), W)
        return varp**0.5  # Minimize volatility
    
    n = len(covariance_matrix)
    W = np.ones([n]) / n  # Initial weights: equally distributed
    bounds = [(0., 1.) for _ in range(n)]  # No short selling
    constraints = [
        {'type': 'eq', 'fun': lambda W: np.sum(W) - 1.}  # Weights sum to 1
    ]
    
    optimized = opt.minimize(
        objective, W, (expected_return, covariance_matrix),
        method='SLSQP', constraints=constraints, bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-08}
    )
    return optimized.x

# Return maximization
def port_maxret(expected_return, covariance_matrix):
    """
    Maximize portfolio return.
    """
    def objective(W, R, C):
        # Portfolio return
        meanp = np.dot(W.T, R)
        return -meanp  # Minimize negative return (maximize return)
    
    n = len(covariance_matrix)
    W = np.ones([n]) / n  # Initial weights: equally distributed
    bounds = [(0., 1.) for _ in range(n)]  # No short selling
    constraints = [
        {'type': 'eq', 'fun': lambda W: np.sum(W) - 1.}  # Weights sum to 1
    ]
    
    optimized = opt.minimize(
        objective, W, (expected_return, covariance_matrix),
        method='SLSQP', constraints=constraints, bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-08}
    )
    return optimized.x
