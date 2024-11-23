def normalize_weights(weights, tolerance=1e-9):
    """
    Normalizes portfolio weights to ensure they sum to exactly 1.0
    
    Parameters:
    weights (numpy.array): Array of portfolio weights
    tolerance (float): Tolerance level for considering weights as valid
    
    Returns:
    numpy.array: Normalized weights that sum to exactly 1.0
    """
    # Check if weights already sum to approximately 1
    total = np.sum(weights)
    if abs(total - 1.0) < tolerance:
        return weights
        
    # Normalize weights
    normalized_weights = weights / total
    
    # Ensure exact sum to 1 by adjusting the largest weight
    sum_diff = 1.0 - np.sum(normalized_weights)
    if abs(sum_diff) > 0:
        max_idx = np.argmax(normalized_weights)
        normalized_weights[max_idx] += sum_diff
        
    # Verify the sum is now exactly 1.0
    assert abs(np.sum(normalized_weights) - 1.0) < tolerance
    
    return normalized_weights

def MIP_robust_ellip(bond_data, lambda_penalty, alpha):

    N = len(bond_data)  # Number of bonds based on the DataFrame length
    W_max = 1.0  # Maximum weight for the portfolio

    # Calculate additional fields
    bond_data['DTS'] = bond_data['ModifiedDuration'] * bond_data['spread']
    bond_data['transaction_cost'] = bond_data['BidAskSpread']
    bond_data['OAS'] = pd.to_numeric(bond_data['spread'], errors='coerce')
    expected_return = np.array(bond_data['ExpectedReturn'])

    # Create the model
    model = gp.Model("MIP_robust_ELIP")

    # Decision variables
    w = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="w")  # weights for bonds
    x = model.addVars(N, vtype=GRB.BINARY, name="x")  # binary selection of bonds

    # Constrain the total weight
    model.addConstr(gp.quicksum(w[i] for i in range(N)) <= W_max, "MaxWeight")
    
    # (DONT REMOVE THIS FROM THE FUNCTION) Ellipsoidal uncertainty set constraint
    epsilon = np.sqrt(chi2.ppf(alpha, df=N)) 
    N_obs = 252  
    theta = np.sqrt(1/N_obs) * np.array(bond_data['StdDev']) 
    t = model.addVar(name="t", vtype=GRB.CONTINUOUS)
    
    # Set the objective: 
    model.setObjective(
        gp.quicksum(expected_return[i] * w[i] - bond_data.iloc[i]['transaction_cost'] * x[i] for i in range(N))
        - lambda_penalty * gp.quicksum(bond_data.iloc[i]['OAS'] * w[i] for i in range(N))
        - lambda_penalty * gp.quicksum(bond_data.iloc[i]['DTS'] * w[i] for i in range(N))
        - epsilon * t,
        GRB.MAXIMIZE
    )
    
    # (DONT REMOVE THIS FROM THE FUNCTION) Ellipsoidal uncertainty set constraint 
    model.addQConstr(
        gp.quicksum((theta[i] * w[i])**2 for i in range(N)) <= t*t,
        "SOC"
    )
        
    # deviation from benchmark constraint
    weights = np.array([1/N] * N)  # Creates array of 75 elements each with 1/75
    normalized_weights = normalize_weights(weights)
    BenchmarkWeight = normalized_weights[0]  # Since all weights are equal, can take any element
    deviation_limit = 0.3
    e = 1e-2  
    for i in range(N):
        model.addConstr(w[i] >= (BenchmarkWeight - deviation_limit*BenchmarkWeight) - e, f"LowerBound_{i}")
        model.addConstr(w[i] <= (BenchmarkWeight + deviation_limit*BenchmarkWeight) + e, f"UpperBound_{i}")

    # sum up to 1 constraint
    e2 = 1e-5
    model.addConstr(gp.quicksum(w[i] for i in range(N)) >= 1 - e2 , "SumToOneLower")
    model.addConstr(gp.quicksum(w[i] for i in range(N)) <= 1 + e2, "SumToOneUpper")

    # OAS Constraints
    weighted_OAS = gp.quicksum(bond_data.iloc[i]['OAS'] * w[i] for i in range(N))
    benchmark_OAS = sum(BenchmarkWeight * bond_data.iloc[i]['OAS'] for i in range(N))  # 1% in each bond
    lower_bound = 0.9 * benchmark_OAS
    upper_bound = 1.1 * benchmark_OAS
    model.addConstr(weighted_OAS >= lower_bound, name="OAS_LowerBound")
    model.addConstr(weighted_OAS <= upper_bound, name="OAS_UpperBound")

    # Liquidity Constraint
    Liquidity = gp.quicksum(bond_data.iloc[i]['LiquidityScore'] for i in range(N)) / N
    MinLiquidity = 0.9 * Liquidity
    model.addConstr(gp.quicksum(bond_data.iloc[i]['LiquidityScore'] * w[i] for i in range(N)) >= MinLiquidity, "MinLiquidity")

    # Transaction Cost Constraints
    Benchmark_cost = gp.quicksum(bond_data.iloc[i]['transaction_cost'] for i in range(N)) / N
    lower_t_cost = 0.9 * Benchmark_cost
    upper_t_cost = 1.1 * Benchmark_cost
    model.addConstr(gp.quicksum(bond_data.iloc[i]['transaction_cost'] * x[i] for i in range(N)) >= lower_t_cost, "MinTCost")
    model.addConstr(gp.quicksum(bond_data.iloc[i]['transaction_cost'] * x[i] for i in range(N)) <= upper_t_cost, "MaxTCost")
    
    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        #print("Optimal solution found. List of all weights:")
        weights = [w[i].X for i in range(N)]  # Get the optimized weights for bonds
        return(np.array(weights))

    if model.status == GRB.INFEASIBLE:
        print("The model is infeasible.")
            