from flask import Flask, request, render_template, jsonify
import pandas as pd
from models.cvar import CVaR_optimization
from models.mvo import mvo_optimize
from models.risk_parity import risk_parity
import numpy as np
from scipy.stats import gmean
import sympy as sp

app = Flask(__name__)

# Endpoint to render the HTML frontend
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to handle form submission
@app.route('/run', methods=['POST'])
def run_optimization():
    try:
        # Extract form data
        model = request.form.get('model')
        additional_constraints = request.form.get('additional_constraints')
        additional_constraints = additional_constraints and eval(additional_constraints)  # Convert JSON to list
        print(model)
        print(additional_constraints)

        #math_constraints = latex_to_math(additional_constraints)
        #print(math_constraints)
        objective_type = request.form.get('objective_type')  # "min" or "max"
        objective_function = request.form.get('objective_function')  # LaTeX string

        print(f"Objective Type: {objective_type}")
        print(f"Objective Function: {objective_function}")
        print(f"Additional Constraints: {additional_constraints}")

        # Handle file upload
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Load CSV
        data = pd.read_csv(file)
        # Placeholder: Process CSV (data validation, etc.)
        
        parsed_constraints = parse_constraints(additional_constraints, data)

        # Run optimization based on model
        result = optimize(model, data, parsed_constraints)
        
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def latex_to_math(latex_constraints):
    """
    Convert LaTeX constraint strings into Python mathematical statements using sympy.
    
    Args:
        latex_constraints (list): A list of constraint strings in LaTeX format.
    
    Returns:
        list: A list of sympy expressions or equations representing the constraints.
    """
    math_constraints = []
    for latex in latex_constraints:
        try:
            # Clean LaTeX syntax for conversion
            cleaned_latex = latex.replace(r"\leq", "<=").replace(r"\geq", ">=").replace(r"\\", "")
            
            # Use sympy to parse the mathematical expression
            constraint = sp.sympify(cleaned_latex, evaluate=False)
            math_constraints.append(constraint)
        except Exception as e:
            # Handle conversion errors
            print(f"Error parsing LaTeX constraint: {latex}\n{e}")
            continue
    
    return math_constraints

def parse_constraints(constraints, data):
    # Function to parse string constraints into usable forms
    parsed = []
    standard_columns = {"expected_return", "OAS", "sector", "duration", "liquidity_score"}
    additional_columns = set(data.columns) - standard_columns  # Capture additional columns

    for constraint in constraints:
        # Ensure constraints reference only columns in data
        if any(col in constraint for col in additional_columns | standard_columns):
            parsed.append(constraint)
        else:
            print(f"Warning: Ignoring constraint '{constraint}' as it references undefined columns.")
    return parsed

# Example optimization function
def optimize(data, model, constraints):
    # Check if model is valid and call the corresponding optimization function
    data = data.sort_values(by='Date')
    data['Date'] = pd.to_datetime(data['Date'])

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

    rf = 0.02 / 365

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

    excess_returns = portfolio_returns - rf

    # Calculate Sharpe ratio
    SR = (gmean(excess_returns + 1) - 1) / excess_returns.std()

    # Average turnover and cumulative transaction cost
    avgTurnover = np.mean(turnover[1:])
    total_transaction_cost = np.sum(transaction_costs)

    print('Sharpe ratio: ', str(SR))
    print('Avg. turnover: ', str(avgTurnover))
    print('Total transaction costs: ', str(total_transaction_cost))

    return SR, avgTurnover, total_transaction_cost


# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
