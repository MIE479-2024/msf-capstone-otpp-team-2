import gurobipy as GRB
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import GRB, quicksum



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


def MIP(bond_data, lambda_penalty, variables, obj, constraints):
    # Create the model
    N = len(bond_data)  # Number of bonds based on the DataFrame length
    W_max = 1.0  # Maximum weight for the portfolio

    
    

    # Create the model
    model = gp.Model("MIP")


    # Decision variables
    w = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="w")  # weights for bonds
    x = model.addVars(N, vtype=GRB.BINARY, name="x")  # binary selection of bonds
    

    # Set the objective: 
    expected_return = np.array(bond_data['ExpectedReturn'])
    model.setObjective(
        gp.quicksum((expected_return[i] * w[i] - bond_data.iloc[i]['transaction_cost'] * x[i]) for i in range(N))
        - lambda_penalty * gp.quicksum(bond_data.iloc[i]['OAS'] * w[i] for i in range(N))
        - lambda_penalty * gp.quicksum(bond_data.iloc[i]['DTS'] * w[i] for i in range(N)),
        GRB.MAXIMIZE,
    )

    if constraints:
        for func in constraints:
            func(model, w, x, bond_data)


    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        #print("Optimal solution found. List of all weights:")
        weights = [w[i].X for i in range(N)]  # Get the optimized weights for bonds
        return np.array(weights)

    if model.status == GRB.INFEASIBLE:
        print("The model is infeasible.")
