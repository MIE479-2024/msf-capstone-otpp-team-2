def MIP_robust_box(bond_data, lambda_penalty, epilson):

    N = len(bond_data)  # Number of bonds based on the DataFrame length

    # Calculate additional fields
    bond_data['DTS'] = bond_data['ModifiedDuration'] * bond_data['spread']
    bond_data['transaction_cost'] = bond_data['BidAskSpread']
    bond_data['OAS'] = pd.to_numeric(bond_data['spread'], errors='coerce')
    expected_return = np.array(bond_data['ExpectedReturn'])

    # Create the model
    model = gp.Model("MIP_robust_box")

    # Decision variables
    w = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="w")  # weights for bonds
    x = model.addVars(N, vtype=GRB.BINARY, name="x")  # binary selection of bonds
    y = model.addVars(N, vtype=GRB.CONTINUOUS, name="y")
    
    # Set the objective: 
    model.setObjective(
        gp.quicksum(expected_return[i] * w[i] - bond_data.iloc[i]['transaction_cost'] * x[i] for i in range(N))
        - lambda_penalty * gp.quicksum(bond_data.iloc[i]['OAS'] * w[i] for i in range(N))
        - lambda_penalty * gp.quicksum(bond_data.iloc[i]['DTS'] * w[i] for i in range(N))
        - epilson * gp.quicksum(bond_data.iloc[i]['StdDev'] * y[i] for i in range(N)),
        GRB.MAXIMIZE
    )
    
    ############# Box Robust constraints (LEAVE THIS IN THIS FUNCTION)
    for i in range(N):
        model.addConstr(y[i] >= w[i], name=f'box_uncertainty_lower_{i}')
        model.addConstr(y[i] >= -w[i], name=f'box_uncertainty_upper_{i}') 
        
    
    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        #print("Optimal solution found. List of all weights:")
        weights = [w[i].X for i in range(N)]  # Get the optimized weights for bonds
        return(np.array(weights))

    if model.status == GRB.INFEASIBLE:
        print("The model is infeasible.")
