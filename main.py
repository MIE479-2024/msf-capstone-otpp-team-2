import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean


bond_data = pd.read_csv('data.csv')
bond_data['Date'] = pd.to_datetime(bond_data['Date'])

# Initial budget to invest ($100,000)
initialVal = 100000  

# Length of investment period (in months)
investPeriod = 1

#Identify the tickers and the dates
tickers = bond_data['ISIN'].unique()
dates = bond_data['Date'].unique()

n = len(tickers)   # Number of bonds
NoPeriods = len(dates) // investPeriod  
# Preallocate space for portfolio weights (x0 will track turnover)
x = np.zeros([n, NoPeriods])
x0 = np.zeros([n, NoPeriods])

# Preallocate space for portfolio value and turnover
currentVal = np.zeros([NoPeriods + 1, 1])
currentVal[0] = initialVal
portfolio_returns = np.zeros(NoPeriods)
turnover = np.zeros([NoPeriods, 1])

rf = 0.02/365

# Iterate through investment periods
for period in range(NoPeriods):
    # Select bonds available during this period
    current_bonds = bond_data[bond_data['Date'] == dates[period * investPeriod]]

    #placeholder
    weights = np.ones(len(current_bonds)) / len(current_bonds)
    
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
    x0[:, period] = weights



excess_returns = portfolio_returns - rf

print(excess_returns.shape)
SR = (gmean(excess_returns + 1) - 1)/excess_returns.std()

avgTurnover = np.mean(turnover[1:])
print('Sharpe ratio: ', str(SR))
print('Avg. turnover: ', str(avgTurnover))

# --------------------------------------------------------------------------
# 3.1 Plot portfolio wealth evolution
# --------------------------------------------------------------------------
plt.figure(1)
plt.plot(np.arange(0, NoPeriods + 1), currentVal, label='Portfolio Value')
plt.title('Bond Portfolio Wealth Evolution')
plt.xlabel('Rebalance Period')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.savefig("bond_wealth_evolution.png")
plt.show()

# --------------------------------------------------------------------------
# 3.2 Plot portfolio weights per rebalance period
# --------------------------------------------------------------------------
fig2 = plt.figure(2)
x[x < 0] = 0
weights = pd.DataFrame(x[(x > 0).any(axis=1)], index=tickers[(x > 0).any(axis=1)])
weights.columns = [col + 1 for col in weights.columns]
weights.T.plot.area(title='Bond Portfolio Weights',
                    ylabel='Weights',
                    xlabel='Rebalance Period',
                    figsize=(6, 3),
                    legend=True,
                    stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig("bond_weights.png")
plt.show()
