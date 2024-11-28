def MIP(bond_data, lambda_penalty, previous_weights):
    # Create the model
    N = len(bond_data)  # Number of bonds based on the DataFrame length


    # Calculate additional fields
    bond_data['DTS'] = bond_data['ModifiedDuration'] * bond_data['spread']
    bond_data['transaction_cost'] = bond_data['BidAskSpread']
    bond_data['OAS'] = pd.to_numeric(bond_data['spread'], errors='coerce')
    expected_return = np.array(bond_data['ExpectedReturn'])

    # Create the model
    model = gp.Model("MIP")

    # Decision variables
    w = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="w")  # weights for bonds
    x = model.addVars(N, vtype=GRB.BINARY, name="x")  # binary selection of bonds

    # Set the objective: 
    model.setObjective(
    gp.quicksum((expected_return[i] * w[i] - bond_data.iloc[i]['transaction_cost'] * x[i]) for i in range(N))  # return minus transaction cost
     - lambda_penalty * gp.quicksum(bond_data.iloc[i]['OAS'] * w[i] for i in range(N))  # OAS risk term
     - lambda_penalty * gp.quicksum(bond_data.iloc[i]['DTS'] * w[i] for i in range(N)),  # DTS risk term
    GRB.MAXIMIZE
    )
    
    # # deviation from benchmark constraint
    weights = np.array([1/N] * N)  # Creates array of 75 elements each with 1/75
    normalized_weights = normalize_weights(weights)
    BenchmarkWeight = normalized_weights[0]  # Since all weights are equal, can take any element
    
    deviation_limit = 0.5  # deviation limit is inputted from user
    for i in range(N):
        model.addConstr(w[i] >= (BenchmarkWeight - deviation_limit*BenchmarkWeight), f"LowerBound_{i}")
        model.addConstr(w[i] <= (BenchmarkWeight + deviation_limit*BenchmarkWeight), f"UpperBound_{i}")

    # sum up to 1 constraint
    e2 = 1e-5
    model.addConstr(gp.quicksum(w[i] for i in range(N)) >= 1 - e2 , "SumToOneLower")
    model.addConstr(gp.quicksum(w[i] for i in range(N)) <= 1 + e2, "SumToOneUpper")
    
    # Spread Constraints
    z = model.addVar(name="z_oas_dev") 
    oas_dev = gp.quicksum((w[i] - BenchmarkWeight)*bond_data.iloc[i]['OAS'] for i in range(N))
    spread_dev = 0.3 # spread_dev is inputted from user 
    model.addConstr(z >= oas_dev, "abs_oas_1")
    model.addConstr(z >= -oas_dev, "abs_oas_2")
    model.addConstr(z <= spread_dev, "oas_deviation_bound")
    
    # Liquidity Constraint
    # this number should be benchmark average liquidity, not portfolio average, thus its an input parameter from benchmark data 
    mean_Liquidity = gp.quicksum(bond_data.iloc[i]['LiquidityScore'] for i in range(N)) / N  
    MinLiquidity = mean_Liquidity  # this value is a placeholder, should be inputted by the user
    model.addConstr(gp.quicksum(bond_data.iloc[i]['LiquidityScore'] * w[i] for i in range(N)) >= MinLiquidity, "MinLiquidity")
    
    # Ratings Constraint
    investment_grade_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
    bond_data['is_investment_grade'] = bond_data['Rating'].apply(lambda x: 1 if x in investment_grade_ratings else 0)
    investment_grade_percentage = 0.3   # should be inputted by the user
    investment_grade_weight = gp.quicksum(bond_data.iloc[i]['is_investment_grade'] * w[i] for i in range(N))
    model.addConstr(investment_grade_weight >= investment_grade_percentage * gp.quicksum(w[i] for i in range(N)), "InvestmentGradeConstraint")
    
    # Turnover Constraint
    max_turnover = 0.5  # should be inputted by the user
    # Variables for turnover calculation
    if previous_weights is not None:
        pos_change = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="pos_change")
        neg_change = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="neg_change")
        
        # Turnover constraints
        for i in range(N):
            # Define relationship between current and previous weights
            model.addConstr(w[i] - previous_weights[i] == pos_change[i] - neg_change[i])
            
        # Total turnover constraint
        model.addConstr(
            quicksum(pos_change[i] + neg_change[i] for i in range(N)) <= 2 * max_turnover,
            "TurnoverLimit"
        )
        
        # Add transaction costs to objective
        transaction_cost_term = quicksum(
            (pos_change[i] + neg_change[i]) * bond_data.iloc[i]['BidAskSpread'] 
            for i in range(N)
        )
    else:
        transaction_cost_term = 0
    
    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        #print("Optimal solution found. List of all weights:")
        weights = [w[i].X for i in range(N)]  # Get the optimized weights for bonds
        return np.array(weights)

    if model.status == GRB.INFEASIBLE:
        print("The model is infeasible.")
