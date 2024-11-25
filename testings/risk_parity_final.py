def risk_parity(bond_data):

    # Calculate transaction cost, daily return, and DTS
    bond_data['transaction_cost'] = bond_data['BidAskSpread']
    bond_data['daily_return'] = bond_data['ExpectedReturn']
    bond_data['DTS'] = bond_data['ModifiedDuration'] * bond_data['spread']

    # Parameters
    N = len(bond_data)

    # Create Gurobi model
    model = gp.Model("DynamicCorporateBondOptimization")

    # Decision variables: weights (continuous)
    w = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="w")

    # Objective: Minimize the difference in risk contributions
    risk_contributions = [w[i] * bond_data.iloc[i]['DTS'] for i in range(N)]
    mean_risk_contribution = gp.quicksum(risk_contributions) / N
    abs_deviations = [model.addVar() for i in range(N)]

    # Constraints to ensure abs_deviations[i] >= |risk_contributions[i] - mean_risk_contribution|
    for i in range(N):
        model.addConstr(abs_deviations[i] >= risk_contributions[i] - mean_risk_contribution)
        model.addConstr(abs_deviations[i] >= mean_risk_contribution - risk_contributions[i])

    # Objective: minimize the sum of the auxiliary absolute deviation variables
    objective = gp.quicksum(abs_deviations[i] for i in range(N))
    model.setObjective(objective, GRB.MINIMIZE)

    # Create array of equal weights first
    weights = np.array([1/75] * 75)  # Creates array of 75 elements each with 1/75

    # Normalize to ensure they sum to exactly 1
    normalized_weights = normalize_weights(weights)

    # Now use the normalized weight as benchmark
    BenchmarkWeight = normalized_weights[0]  # Since all weights are equal, can take any element

    # 4. Weight sum constraint: sum of weights must be 1
    model.addConstr(gp.quicksum(w[i] for i in range(N)) >= 0.99 , "SumToOneLower")
    model.addConstr(gp.quicksum(w[i] for i in range(N)) <= 1, "SumToOneUpper")

    # Optimize the Gurobi model
    model.optimize()

    # Recheck feasibility
    status = model.Status
    if status in [GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED]:
        print("The relaxed model is still infeasible or unbounded.")
        sys.exit(1)
    elif status != GRB.OPTIMAL:
        print(f"Optimization was stopped with status {status}.")
        sys.exit(1)

    # Print slack values
    print("\nSlack values:")
    orignumvars = model.NumVars
    slacks = model.getVars()[orignumvars:]
    for sv in slacks:
        print(f"{sv.VarName} = {sv.X:g}")

    return np.array([w[i].X for i in range(N)])