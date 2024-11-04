import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import GRB, quicksum

def CVaR_optimization(bond_data, alpha, constraints):
    
    N = len(bond_data)  # Number of bonds based on the DataFrame length
    W_max = 1.0  # Maximum weight for the portfolio
    M = W_max  # Maximum weight for any single bond

    # Calculate additional fields
    bond_data['DTS'] = bond_data['duration'] * bond_data['OAS']
    bond_data['transaction_cost'] = bond_data['ask_price'] - bond_data['bid_price']
    expected_returns = np.array(bond_data["expected_return"])

    # Create the model
    model = gp.Model("CVaR_Bond_Optimization")

    # Decision variables
    w = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="w")  # weights for bonds
    x = model.addVars(N, vtype=GRB.BINARY, name="x")  # binary selection of bonds
    VaR = model.addVar(vtype=GRB.CONTINUOUS, name="VaR")  # Value at Risk variable
    z = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="z")  # Auxiliary for CVaR

    # Link weights w[i] with selection x[i]
    for i in range(N):
        model.addConstr(w[i] <= x[i] * M, "WeightSelection_" + str(i))

    # Constrain the total weight
    model.addConstr(quicksum(w[i] for i in range(N)) <= W_max, "MaxWeight")
    model.addConstr(quicksum(w[i] for i in range(N)) == 1, "WeightSum")

    # VaR constraint
    model.addConstr(VaR >= quicksum(w[i] * expected_returns[i] for i in range(N)), "VaRDef")

    # CVaR auxiliary constraints
    for i in range(N):
        model.addConstr(z[i] >= w[i] - VaR, "CVaR_Aux_" + str(i))

    # Objective function: Minimize CVaR risk
    model.setObjective(
        VaR + (1 / (N * (1 - alpha))) * quicksum(z[i] for i in range(N)),
        GRB.MINIMIZE
    )

    # OAS Constraints
    benchmark_OAS = 274.88
    lower_bound = 0.9 * benchmark_OAS
    upper_bound = 1.1 * benchmark_OAS

    model.addConstr(quicksum(bond_data.iloc[i]['OAS'] * w[i] for i in range(N)) >= lower_bound, "MinOAS")
    model.addConstr(quicksum(bond_data.iloc[i]['OAS'] * w[i] for i in range(N)) <= upper_bound, "MaxOAS")

    # Liquidity Constraint
    Liquidity = quicksum(bond_data.iloc[i]['liquidity_score'] for i in range(N)) / N
    MinLiquidity = 0.9 * Liquidity
    model.addConstr(quicksum(bond_data.iloc[i]['liquidity_score'] * w[i] for i in range(N)) >= MinLiquidity, "MinLiquidity")

    # Transaction Cost Constraints
    Benchmark_cost = quicksum(bond_data.iloc[i]['transaction_cost'] for i in range(N)) / N
    lower_t_cost = 0.9 * Benchmark_cost
    upper_t_cost = 1.1 * Benchmark_cost

    model.addConstr(quicksum(bond_data.iloc[i]['transaction_cost'] * x[i] for i in range(N)) >= lower_t_cost, "MinTCost")
    model.addConstr(quicksum(bond_data.iloc[i]['transaction_cost'] * x[i] for i in range(N)) <= upper_t_cost, "MaxTCost")

    # additional constraints
    for constraint in constraints:
        if constraint['operator'] == '<=':
            model.addConstr(gp.quicksum(bond_data.loc[i, constraint['column']] for i in range(N)) <= constraint['value'], "<=")
        elif constraint['operator'] == '<':
            model.addConstr(-gp.quicksum(bond_data.loc[i, constraint['column']] for i in range(N)) >= -constraint['value'], "<")
        elif constraint['operator'] == '>=':
            model.addConstr(gp.quicksum(bond_data.loc[i, constraint['column']] for i in range(N)) >= constraint['value'], ">=")
        elif constraint['operator'] == '>':
            model.addConstr(-gp.quicksum(bond_data.loc[i, constraint['column']] for i in range(N)) <= -constraint['value'], ">")
        elif constraint['operator'] == '=':
            model.addConstr(gp.quicksum(bond_data.loc[i, constraint['column']] for i in range(N)) <= constraint['value'], "=1")
            model.addConstr(gp.quicksum(bond_data.loc[i, constraint['column']] for i in range(N)) >= constraint['value'], "=2")



    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("Optimal solution found. List of all weights:")
        weights = [w[i].X for i in range(N)]  # Get the optimized weights for bonds
        return np.array(weights)

    else:
        print("No optimal solution found.")
        return None
