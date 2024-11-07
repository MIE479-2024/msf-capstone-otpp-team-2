def risk_parity(bond_data, original_weights):

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

    # Slack variables for relaxation of constraints
    slack_liquidity = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="slack_liquidity")
    slack_tcost_lower = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="slack_tcost_lower")
    slack_tcost_upper = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="slack_tcost_upper")

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
    benchmark_OAS = 274.88
    lower_bound = 0.9 * benchmark_OAS
    upper_bound = 1.1 * benchmark_OAS
    model.addConstr(gp.quicksum(bond_data.loc[i, 'OAS'] * w[i] for i in range(N)) >= lower_bound, "MinOAS")
    model.addConstr(gp.quicksum(bond_data.loc[i, 'OAS'] * w[i] for i in range(N)) <= upper_bound, "MaxOAS")

    # 2. Liquidity constraint: portfolio liquidity must be at least 90% of benchmark liquidity (with relaxation)
    Liquidity = gp.quicksum(bond_data.loc[i, 'liquidity_score'] * BenchmarkWeight for i in range(N))
    MinLiquidity = 0.9 * Liquidity
    model.addConstr(gp.quicksum(bond_data.loc[i, 'liquidity_score'] * w[i] for i in range(N)) >= MinLiquidity -
                     slack_liquidity, "MinLiquidity")

    # 3. Transaction cost constraint: within 10% of benchmark transaction cost (with relaxation)
    Benchmark_cost = gp.quicksum(bond_data.loc[i, 'transaction_cost'] * BenchmarkWeight for i in range(N))
    lower_t_cost = 0.9 * Benchmark_cost
    upper_t_cost = 1.1 * Benchmark_cost
    model.addConstr(gp.quicksum(bond_data.loc[i, 'transaction_cost'] for i in range(N)) >= lower_t_cost -
    slack_tcost_lower, "MintCost")

    model.addConstr(gp.quicksum(bond_data.loc[i, 'transaction_cost'] for i in range(N)) <= upper_t_cost +
    slack_tcost_upper, "MaxtCost")

    # 4. Weight sum constraint: sum of weights must be 1
    model.addConstr(gp.quicksum(w[i] for i in range(N)) == 1, "WeightSum")

    # 6. No sector can exceed 40% of portfolio weight
    sectors = bond_data['sector'].unique()

    for sector in sectors:
        sector_bonds = bond_data[bond_data['sector'] == sector].index
        model.addConstr(gp.quicksum(w[i] for i in sector_bonds) <= 0.4, f"MaxSectorWeight_{sector}")


    # 7. New weight constraint: Each bond's weight must be within 5% of the benchmark weight

    slack_weight_lower = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="slack_weight_lower")
    slack_weight_upper = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="slack_weight_upper")

    for i in range(N):
        model.addConstr(w[i] >= 0.95 * BenchmarkWeight - slack_weight_lower[i], f"MinWeight_{i}")
        model.addConstr(w[i] <= 1.05 * BenchmarkWeight + slack_weight_upper[i], f"MaxWeight_{i}")

    penalty_weight = 1000  # Penalty for violating weight constraints
    penalty = 1000
    model.setObjective(objective + penalty * (slack_liquidity + slack_tcost_lower + slack_tcost_upper) +
                   penalty_weight * gp.quicksum(slack_weight_lower[i] + slack_weight_upper[i] for i in range(N)),
                   GRB.MINIMIZE)

    # Optimize the Gurobi model
    model.optimize()

    # Check if the solution is feasible or infeasible, and return the results
    if model.status == GRB.OPTIMAL:
        # Retrieve optimized weights
        new_weights = [w[i].x for i in range(N)]
        return new_weights
    else:
        raise Exception("Optimization did not yield an optimal solution.")

