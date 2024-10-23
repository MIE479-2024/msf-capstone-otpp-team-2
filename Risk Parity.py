import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load data and process it by grouping rows into daily batches of bonds
def get_daily_data(file_path, bonds_per_day=100):
    df = pd.read_csv(file_path)

    # Add a new column representing the 'Date Group'
    df['Date_Group'] = np.repeat(np.arange(len(df) // bonds_per_day), bonds_per_day)

    # Group the data by 'Date_Group', so each group contains data for 100 bonds per day
    grouped = df.groupby('Date_Group')
    return grouped, df['ISIN'].unique()

# Turnover constraint function to limit daily weight changes
def turnover_constraint(weights_new, weights_prev, max_change):
    change = np.abs(weights_new - weights_prev)
    return max_change - change

# Risk Parity Optimization Functions
def portfolio_risk(weights, dts_values):
    weighted_dts = weights * dts_values
    return np.sqrt(np.sum(weighted_dts ** 2))

def risk_contribution(weights, dts_values):
    total_portfolio_risk = portfolio_risk(weights, dts_values)
    weighted_dts = weights * dts_values
    marginal_contribution = weighted_dts / total_portfolio_risk
    return marginal_contribution * weights

def risk_parity_objective(weights, dts_values):
    risk_contrib = risk_contribution(weights, dts_values)
    avg_contrib = np.mean(risk_contrib)
    return np.sum((risk_contrib - avg_contrib) ** 2)

# Weights must sum to 1
def weight_sum_constraint(weights):
    return np.sum(weights) - 1

# Boundaries for the weights: between 0 and 1
def get_bounds(num_assets):
    return [(0, 1) for _ in range(num_assets)]

# Assume there are 100 bonds and each day has 100 rows in sequence for the bonds
bonds_per_day = 100
filepath = "E:\\4F\\Capstone\\bond_data_rev1.csv"

# Fetch the grouped data for each day and the unique ISINs
daily_data_groups, unique_isins = get_daily_data(filepath, bonds_per_day)

# Initialize a list to store the portfolio weights for each day
all_weights = []

# Set initial weights for the first day (equal weights)
initial_weights = np.ones(bonds_per_day) / bonds_per_day

# Define the maximum allowable change in weights (10%)
max_change = 0.10

# Initialize a variable to accumulate turnover
total_turnover = 0
turnover_count = 0

# Iterate over each day's data and perform optimization
for day, group in daily_data_groups:
    # Extract OAS, Duration, and Standard Deviation for all bonds for this day
    mu = group['OAS'].to_numpy() / 100  # Use OAS as expected return, divided by 100
    duration = group['duration'].to_numpy()
    std_dev = group['return_std_dev'].to_numpy()

    # Filter out NaNs (if any)
    valid_indices = ~np.isnan(mu) & ~np.isnan(duration) & ~np.isnan(std_dev)
    mu_filtered = mu[valid_indices]
    duration_filtered = duration[valid_indices]
    std_dev_filtered = std_dev[valid_indices]

    number_of_assets = len(mu_filtered)

    if number_of_assets == 0:
        print(f"No valid data for day {day + 1}")
        continue

    # Initial weights (x0) are the same as the previous day's weights
    if day == 0:
        x0 = initial_weights
    else:
        x0 = all_weights[-1]

    # Calculate DTS (Duration Times Spread)
    DTS = duration_filtered * mu_filtered

    # Run Risk Parity Optimization
    constraints = [
        {'type': 'eq', 'fun': weight_sum_constraint},  # Weights must sum to 1
        {'type': 'ineq', 'fun': lambda w: turnover_constraint(w, x0, max_change)}  # Turnover constraint
    ]
    bounds = get_bounds(number_of_assets)

    result = minimize(risk_parity_objective, x0, args=(DTS,), method='SLSQP', bounds=bounds, constraints=constraints)

    # Store optimized weights for this day
    all_weights.append(result.x)

    # Calculate turnover if not the first day
    if day > 0:
        turnover = np.sum(np.abs(result.x - all_weights[-2]))
        total_turnover += turnover
        turnover_count += 1

# Convert the list of weights into a numpy array
all_weights = np.array(all_weights)

# Compute the average portfolio weights over all days
average_weights = np.mean(all_weights, axis=0)

# Calculate the expected portfolio return using the OAS values
average_return = np.dot(average_weights, mu_filtered)
portfolio_std_dev = np.sqrt(np.dot(average_weights**2, std_dev_filtered**2))

# Sharpe ratio (assuming given standard deviation is for excess return)
sharpe_ratio = average_return / portfolio_std_dev if portfolio_std_dev != 0 else np.nan

# Calculate averaged turnover
averaged_turnover = total_turnover / turnover_count if turnover_count > 0 else 0

# Print the final average portfolio weights and their corresponding ISINs
print("Averaged Optimized Weights for Each Bond:")
for isin, weight in zip(unique_isins[:bonds_per_day], average_weights):
    print(f"ISIN: {isin}, Weight: {weight}")

# Print the Sharpe ratio and averaged turnover
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Averaged Turnover: {averaged_turnover}")


