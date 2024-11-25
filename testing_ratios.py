import numpy as np
from scipy.stats import gmean

def max_drawdown(weights_matrix, prices_matrix):
    #weights should be a NxT matrix where each column is a vector of weights for each time period
    portfolio_values = np.dot(prices_matrix.T, weights_matrix)
    portfolio_values = portfolio_values.sum(axis = 1)

    max_value = np.max(portfolio_values)
    max_index = np.argmax(portfolio_values)
    portfolio_values = portfolio_values[max_index:]
    min_value = np.min(portfolio_values)
    
    #Calculate max drawdown
    
    mdd = (max_value - min_value)/max_value
    return mdd

def treynor_ratio(portfolio_returns, asset_returns):
    #Calculate the returns for the "market"
    market_returns = np.mean(asset_returns, axis=0)  

    #Calculate the covariance between the "market" and the portfolio
    cov_matrix = np.cov(portfolio_returns, market_returns)
    cov_portfolio_market = cov_matrix[0, 1]  # Covariance between portfolio and market returns
    var_market = np.var(market_returns)  # Variance of the market returns

    # Beta calculation
    portfolio_beta = cov_portfolio_market / var_market

    #calculate the treynor ratio
    portfolio_mean_return = gmean(1 + portfolio_returns) - 1
    treynor_ratio = portfolio_mean_return/ portfolio_beta
    return treynor_ratio