import pandas as pd
from models.cvar import CVaR_optimization
from models.mvo import mvo_optimize
from models.risk_parity import risk_parity
import numpy as np
from scipy.stats import gmean
import sympy as sp


data = pd.read_csv('data.csv')
# Check if model is valid and call the corresponding optimization function
data = data.sort_values(by='Date')
data['Date'] = pd.to_datetime(data['Date'])

model = 'cVaR'
constraints = []
# Initial budget to invest ($100,000)
initialVal = 100000  

# Length of investment period 
investPeriod = 1

# Identify the tickers and the dates
tickers = data['ISIN'].unique()
dates = data['Date'].unique()

n = len(tickers)   # Number of bonds
NoPeriods = len(dates) // investPeriod  

# Preallocate space for portfolio weights (x0 will track turnover)
x = np.zeros([n, NoPeriods])
x0 = np.zeros([n, NoPeriods])

# Preallocate space for portfolio value, turnover, and transaction costs
currentVal = np.zeros([NoPeriods + 1, 1])
currentVal[0] = initialVal
portfolio_returns = np.zeros(NoPeriods)
turnover = np.zeros([NoPeriods, 1])
transaction_costs = np.zeros([NoPeriods, 1])

# Iterate through investment periods
for period in range(NoPeriods):
    # Select bonds available during this period
    current_bonds = data[data['Date'] == dates[period * investPeriod]]


    
    if model == "MVO":
        weights = mvo_optimize(data, x0, constraints)
    elif model == "Risk Parity":
        weights = risk_parity(data, x0, constraints)
    elif model == "cVaR":
        weights = CVaR_optimization(data, 0.95, 10, constraints)
    else:
        print("Invalid model selected.")
    
    # Store weights
    x[:, period] = weights
    
    # Calculate expected portfolio return and risk based on the bonds
    portfolio_return = np.sum(weights * current_bonds['expected_return'])
    portfolio_risk = np.sqrt(np.sum((weights**2) * (current_bonds['return_std_dev']**2)))
    
    # Simulate the portfolio value evolution
    currentVal[period + 1] = currentVal[period] * (1 + portfolio_return)
    portfolio_returns[period] = portfolio_return

    # Calculate turnover (assume full rebalance between periods)
    turnover[period] = np.sum(np.abs(weights - x0[:, period])) / 2

    # Calculate transaction costs
    turnover_weights = np.abs(weights - x0[:, period])

    transaction_costs[period] = np.sum(turnover_weights * current_bonds['BidAskSpread'].values)
    
    # Adjust current value for transaction costs
    currentVal[period + 1] -= transaction_costs[period]

    # Update previous weights for next period
    x0[:, period] = weights

excess_returns = portfolio_returns

# Calculate Sharpe ratio
SR = (gmean(excess_returns + 1) - 1) / excess_returns.std()

# Average turnover and cumulative transaction cost
avgTurnover = np.mean(turnover[1:])
total_transaction_cost = np.sum(transaction_costs)

print('Sharpe ratio: ', str(SR))
print('Avg. turnover: ', str(avgTurnover))
print('Total transaction costs: ', str(total_transaction_cost))

