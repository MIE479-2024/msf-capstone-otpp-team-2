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
    objective = gp.quicksum((risk_contributions[i] - mean_risk_contribution) ** 2 for i in range(N))

    # Set the model objective to minimize the variance of risk contributions
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
    model.addConstr(gp.quicksum(bond_data.loc[i, 'liquidity_score'] * w[i] for i in range(N)) >= MinLiquidity - slack_liquidity, "MinLiquidity")

    # 3. Transaction cost constraint: within 10% of benchmark transaction cost (with relaxation)
    Benchmark_cost = gp.quicksum(bond_data.loc[i, 'transaction_cost'] * BenchmarkWeight for i in range(N))
    lower_t_cost = 0.9 * Benchmark_cost
    upper_t_cost = 1.1 * Benchmark_cost
    model.addConstr(gp.quicksum(bond_data.loc[i, 'transaction_cost'] for i in range(N)) >= lower_t_cost - slack_tcost_lower, "MintCost")
    model.addConstr(gp.quicksum(bond_data.loc[i, 'transaction_cost'] for i in range(N)) <= upper_t_cost + slack_tcost_upper, "MaxtCost")

    # 4. Weight sum constraint: sum of weights must be 1
    model.addConstr(gp.quicksum(w[i] for i in range(N)) == 1, "WeightSum")

    # Set slack variables' penalty in the objective function
    penalty = 1000  # A large value to minimize slack
    model.setObjective(objective + penalty * (slack_liquidity + slack_tcost_lower + slack_tcost_upper), GRB.MINIMIZE)

    # Optimize the Gurobi model
    model.optimize()

    # Check if the solution is feasible or infeasible, and return the results
    if model.status == GRB.OPTIMAL:
        # Retrieve optimized weights
        new_weights = [w[i].x for i in range(N)]
        return new_weights
    else:
        raise Exception("Optimization did not yield an optimal solution.")

