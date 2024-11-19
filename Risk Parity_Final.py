def risk_parity(bond_data, original_weights):
    import gurobipy as gp
    from gurobipy import GRB

    BenchmarkWeight = 0.01

    # Calculate transaction cost, daily return, and DTS
    bond_data['transaction_cost'] = bond_data['ask_price'] - bond_data['bid_price']
    bond_data['daily_return'] = bond_data['expected_return']
    bond_data['DTS'] = bond_data['duration'] * bond_data['OAS']

    # Parameters
    N = 100  # Number of bonds

    # Create Gurobi model
    model = gp.Model("DynamicCorporateBondOptimization")

    # Decision variables: weights (continuous)
    w = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="w")

    # Objective: Minimize the difference in risk contributions
    risk_contributions = [w[i] * bond_data.loc[i, 'DTS'] for i in range(N)]
    mean_risk_contribution = gp.quicksum(risk_contributions) / N
    abs_deviations = [model.addVar() for i in range(N)]

    # Constraints to ensure abs_deviations[i] >= |risk_contributions[i] - mean_risk_contribution|
    for i in range(N):
        model.addConstr(abs_deviations[i] >= risk_contributions[i] - mean_risk_contribution)
        model.addConstr(abs_deviations[i] >= mean_risk_contribution - risk_contributions[i])

    # Objective: minimize the sum of the auxiliary absolute deviation variables
    objective = gp.quicksum(abs_deviations[i] for i in range(N))
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints

    # 1. OAS risk constraints: within 10% of benchmark OAS
    benchmark_OAS = BenchmarkWeight * gp.quicksum(bond_data.loc[i, 'OAS'] for i in range(N))
    model.addConstr(gp.quicksum(bond_data.loc[i, 'OAS'] * w[i] for i in range(N)) >= 0.9 * benchmark_OAS, "MinOAS")
    model.addConstr(gp.quicksum(bond_data.loc[i, 'OAS'] * w[i] for i in range(N)) <= 1.1 * benchmark_OAS, "MaxOAS")

    # 2. Liquidity constraint: portfolio liquidity must be at least 90% of benchmark liquidity
    benchmark_liquidity = BenchmarkWeight * gp.quicksum(bond_data.loc[i, 'liquidity_score'] for i in range(N))
    model.addConstr(gp.quicksum(bond_data.loc[i, 'liquidity_score'] * w[i] for i in range(N)) >= 0.9 * benchmark_liquidity, "MinLiquidity")

    # 3. Transaction cost constraint: within 10% of benchmark transaction cost
    benchmark_cost = BenchmarkWeight * gp.quicksum(bond_data.loc[i, 'transaction_cost'] for i in range(N))
    model.addConstr(gp.quicksum(bond_data.loc[i, 'transaction_cost'] * w[i] for i in range(N)) >= 0.9 * benchmark_cost, "MinTCost")
    model.addConstr(gp.quicksum(bond_data.loc[i, 'transaction_cost'] * w[i] for i in range(N)) <= 1.1 * benchmark_cost, "MaxTCost")

    # 4. Weight sum constraint: sum of weights must be 1
    model.addConstr(gp.quicksum(w[i] for i in range(N)) == 1, "WeightSum")

    # 5. No sector can exceed 40% of portfolio weight
    sectors = bond_data['sector'].unique()
    for sector in sectors:
        sector_bonds = bond_data[bond_data['sector'] == sector].index
        model.addConstr(gp.quicksum(w[i] for i in sector_bonds) <= 0.4, f"MaxSectorWeight_{sector}")

    # 6. New weight constraint: Each bond's weight must be within 5% of the benchmark weight
    for i in range(N):
        model.addConstr(w[i] >= 0.95 * BenchmarkWeight, f"MinWeight_{i}")
        model.addConstr(w[i] <= 1.05 * BenchmarkWeight, f"MaxWeight_{i}")

    # Optimize the Gurobi model
    model.optimize()

    # Check if the solution is feasible or infeasible, and return the results
    if model.status == GRB.OPTIMAL:
        # Retrieve optimized weights
        new_weights = [w[i].x for i in range(N)]
        return new_weights
    else:
        raise Exception("Optimization did not yield an optimal solution.")
