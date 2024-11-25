import pandas as pd
from models.cvar import CVaR_optimization
from models.mvo import mvo_optimize
from models.risk_parity import risk_parity
import numpy as np
from scipy.stats import gmean
import sympy as sp
from models.mip import MIP
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify
from sympy.parsing.latex import parse_latex
from sympy import Sum, Eq, Le, Ge
from sympy.core.symbol import Symbol
from sympy.functions import Piecewise
from gurobipy import Model, GRB, quicksum
from models.mip import normalize_weights
import gurobipy as gp
from gurobipy import GRB
from gurobipy import GRB, quicksum

# Initial setup remains the same
pd.set_option('mode.chained_assignment', None)


app = Flask(__name__)


def std_constraints(model):
    constraints = []
    if model == 'MVO':
        constraints.append(parse_latex(" \\sum_{i=1}^{N} x_i \\leq 1 "))

    print(constraints)

def sympy_to_gurobi(expr, gurobi_model, gurobi_vars):
    """
    Converts a Sympy parsed constraint into a Gurobi constraint and adds it to the model.

    :param expr: Sympy parsed expression (e.g., Eq(Sum(x[i], (i, 1, N)), 1)).
                 Handles Eq (equality), Le (<=), Ge (>=).
    :param gurobi_model: Gurobi model instance.
    :param gurobi_vars: Dictionary of Gurobi variables, indexed by their names or other identifiers.
                        e.g., {"x_1": x1, "x_2": x2, ...}
    :return: Added Gurobi constraint.
    """
    # Determine constraint type
    if isinstance(expr, (Eq, Le, Ge)):
        lhs, rhs = expr.lhs, expr.rhs
        comparator = expr.rel_op  # Gets relational operator (e.g., "==" or "<=" or ">=")
    else:
        raise ValueError("Unsupported constraint type. Only Eq, Le, and Ge are supported.")

    # Handle Sympy Sum
    if isinstance(lhs, Sum):
        summand, (index, start, end) = lhs.args
        # Convert the summand into Gurobi variables
        if isinstance(summand, Symbol):  # Assuming simple sums like Sum(x[i], (i, 1, N))
            gurobi_sum = quicksum(
                gurobi_vars[f"{summand.name}_{i}"] for i in range(int(start), int(end) + 1)
            )
        else:
            raise NotImplementedError("Complex summands in Sum are not yet supported.")
    else:
        raise NotImplementedError("Expressions without Sum are not yet supported.")

    # Add the appropriate constraint to the Gurobi model
    if comparator == "==":  # Equality
        gurobi_constraint = gurobi_model.addConstr(gurobi_sum == float(rhs))
    elif comparator == "<=":  # Less than or equal
        gurobi_constraint = gurobi_model.addConstr(gurobi_sum <= float(rhs))
    elif comparator == ">=":  # Greater than or equal
        gurobi_constraint = gurobi_model.addConstr(gurobi_sum >= float(rhs))
    else:
        raise ValueError("Unsupported relational operator: " + comparator)

    return gurobi_constraint


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_portfolio_optimization(initialVal=100000, investPeriod=1):


    uploaded_file = request.files['file']  # Uploaded CSV file
    model = request.form['model']  # Optimization model
    objective_type = request.form.get('objective_type', 'min')  # Minimize/Maximize
    objective_function = request.form.get('objective_function', '')  # Objective function
    additional_constraints = request.form.get('additional_constraints', '[]')  # Additional constraints (JSON)
    additional_constraints = additional_constraints and eval(additional_constraints)  # Convert JSON to list
    objective_function = objective_function[4:]
    objective_function = parse_latex(objective_function)

    for i in range(len(additional_constraints)):
        additional_constraints[i] = parse_latex(additional_constraints[i])


    
    
    
    # Read the uploaded file into a pandas DataFrame
    if not uploaded_file:
        return jsonify({'error': 'No file uploaded'}), 400
    data = pd.read_csv(uploaded_file)

    # VARS---------------------------------------------------------------------------------------------------------------
    def w(model, bond_data):
        N = len(bond_data)
        w = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="w")  # weights for bonds

    def x(model, bond_data):
        N = len(bond_data)
        x = model.addVars(N, vtype=GRB.BINARY, name="x")  # binary selection of bonds

    def var(model, bond_data):
        VaR = model.addVar(vtype=GRB.CONTINUOUS, name="VaR")  # Value at Risk variable
    #def z_var(model, bond_data):
    #    z = model.addVars(S, vtype=GRB.CONTINUOUS, lb=0, name="z")  # Auxiliary for CVaR


    # VARS---------------------------------------------------------------------------------------------------------------
    # CONSTRAINTS---------------------------------------------------------------------------------------------------------------
    def benchmark_dev(model, w, x, bond_data):
        # deviation from benchmark constraint
        N = len(bond_data)
        weights = np.array([1/N] * N)  # Creates array of 75 elements each with 1/75
        normalized_weights = normalize_weights(weights)
        BenchmarkWeight = normalized_weights[0]  # Since all weights are equal, can take any element
        deviation_limit = 0.3
        for i in range(N):
            model.addConstr(w[i] >= (BenchmarkWeight - deviation_limit*BenchmarkWeight), f"LowerBound_{i}")
            model.addConstr(w[i] <= (BenchmarkWeight + deviation_limit*BenchmarkWeight), f"UpperBound_{i}")

    def sum_to_one(model, w, x, bond_data):
        # sum up to 1 constraint
        N = len(bond_data)
        e2 = 1e-5
        model.addConstr(gp.quicksum(w[i] for i in range(N)) >= 1 - e2 , "SumToOneLower")
        model.addConstr(gp.quicksum(w[i] for i in range(N)) <= 1 + e2, "SumToOneUpper")
        
    def OAS(model, w, x, bond_data):
        # OAS Constraints
        N = len(bond_data)
        weights = np.array([1/N] * N)  # Creates array of 75 elements each with 1/75
        normalized_weights = normalize_weights(weights)
        BenchmarkWeight = normalized_weights[0]  # Since all weights are equal, can take any element

        weighted_OAS = gp.quicksum(bond_data.iloc[i]['OAS'] * w[i] for i in range(N))
        benchmark_OAS = sum(BenchmarkWeight * bond_data.iloc[i]['OAS'] for i in range(N))  # 1% in each bond
        lower_bound = 0.9 * benchmark_OAS
        upper_bound = 1.1 * benchmark_OAS

        model.addConstr(weighted_OAS >= lower_bound, name="OAS_LowerBound")
        model.addConstr(weighted_OAS <= upper_bound, name="OAS_UpperBound")

    def liquidity(model, w, x, bond_data):
        N = len(bond_data)
        # Liquidity Constraint
        Liquidity = gp.quicksum(bond_data.iloc[i]['LiquidityScore'] for i in range(N)) / N
        MinLiquidity = 0.9 * Liquidity
        model.addConstr(gp.quicksum(bond_data.iloc[i]['LiquidityScore'] * w[i] for i in range(N)) >= MinLiquidity, "MinLiquidity")

    def t_cost(model, w, x, bond_data):
        N = len(bond_data)
        # Transaction Cost Constraints
        Benchmark_cost = gp.quicksum(bond_data.iloc[i]['transaction_cost'] for i in range(N)) / N
        lower_t_cost = 0.9 * Benchmark_cost
        upper_t_cost = 1.1 * Benchmark_cost

        model.addConstr(gp.quicksum(bond_data.iloc[i]['transaction_cost'] * x[i] for i in range(N)) >= lower_t_cost, "MinTCost")
        model.addConstr(gp.quicksum(bond_data.iloc[i]['transaction_cost'] * x[i] for i in range(N)) <= upper_t_cost, "MaxTCost")
    # CONSTRAINTS---------------------------------------------------------------------------------------------------------------


    try:

        # Identify the tickers and the dates
        tickers = data['SecurityId'].unique()
        dates = data['Date'].unique()

        n = len(tickers)   # Number of bonds
        NoPeriods = len(dates) // investPeriod  

        # Preallocate space for portfolio weights
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
            # Determine the current date
            current_date = dates[period * investPeriod]
            
            # Get current period's bonds and filter for positive expected returns
            current_bonds = data[data['Date'] == current_date]
            positive_return_bonds = current_bonds[current_bonds['ExpectedReturn'] > 0]
            
            # Check if there are enough data points to proceed
            if positive_return_bonds.empty or positive_return_bonds['ExpectedReturn'].isnull().any() or positive_return_bonds['StdDev'].isnull().any():
                print(f"Skipping period {period}: insufficient data after filtering for positive returns.")
                continue

            active_bonds = positive_return_bonds

            #additional fields
            active_bonds['DTS'] = active_bonds['ModifiedDuration'] * active_bonds['spread']
            active_bonds['transaction_cost'] = active_bonds['BidAskSpread']
            active_bonds['OAS'] = pd.to_numeric(active_bonds['spread'], errors='coerce')
            
            # Get optimal weights for filtered bonds
            if model == "MVO":
                weights = mvo_optimize(data, x0, constraints)
            elif model == "Risk Parity":
                weights = risk_parity(data, x0, constraints)
            elif model == "cVaR":
                weights = CVaR_optimization(data, 0.95, 10, constraints)
            elif model == "MIP":
                variables = [x, w]
                obj = "model.setObjective(gp.quicksum((expected_return[i] * w[i] - bond_data.iloc[i]['transaction_cost'] * x[i]) for i in range(N)) - lambda_penalty * gp.quicksum(bond_data.iloc[i]['OAS'] * w[i] for i in range(N)) - lambda_penalty * gp.quicksum(bond_data.iloc[i]['DTS'] * w[i] for i in range(N)), GRB.MAXIMIZE)"
                constraints = [benchmark_dev, sum_to_one, OAS, liquidity, t_cost]
                weights = MIP(active_bonds, 1, variables, obj, constraints)
            else:
                print("Invalid model selected.")
            
            
            if weights is None or not weights.any():
                print(f"Optimization failed for period {period}")
                continue
            
            # Map the weights back to the full universe (with zeros for filtered out bonds)
            full_weights = np.zeros(n)
            for i, security_id in enumerate(positive_return_bonds['SecurityId']):
                idx = list(tickers).index(security_id)
                full_weights[idx] = weights[i]
            
            # Store weights
            x[:, period] = full_weights
            
            # Portfolio calculations
            portfolio_return = np.sum(full_weights * current_bonds['ExpectedReturn'])
            currentVal[period + 1] = currentVal[period] + portfolio_return
            portfolio_returns[period] = portfolio_return

            # Turnover and transaction costs
            turnover[period] = np.sum(np.abs(full_weights - x0[:, period])) / 2
            turnover_weights = np.abs(full_weights - x0[:, period])
            transaction_costs[period] = np.sum(turnover_weights * current_bonds['BidAskSpread'].values)
            currentVal[period + 1] -= transaction_costs[period]
            x0[:, period] = full_weights

        # Calculate performance metrics
        excess_returns = portfolio_returns
        SR = (gmean(excess_returns + 1) - 1) / excess_returns.std()
        avgTurnover = np.mean(turnover[1:])
        total_transaction_cost = np.sum(transaction_costs)

        # Print metrics
        print('Sharpe ratio: ', str(SR))
        print('Avg. turnover: ', str(avgTurnover))
        print('Total transaction costs: ', str(total_transaction_cost))

        # Plot portfolio wealth evolution
        plt.figure(1)
        plt.plot(np.arange(0, NoPeriods + 1), currentVal, label='Portfolio Value')
        plt.title('Bond Portfolio Wealth Evolution')
        plt.xlabel('Rebalance Period')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.savefig("bond_wealth_evolution.png")
        plt.show()
        #print(currentVal)

        # Plot portfolio weights
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

        return {
            'portfolio_values': currentVal,
            'portfolio_returns': portfolio_returns,
            'weights': x,
            'turnover': turnover,
            'transaction_costs': transaction_costs
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the optimization
start_date = pd.Timestamp('2025-01-01')  
end_date = pd.Timestamp('2025-06-18')    

if __name__ == '__main__':
    app.run(debug=True)