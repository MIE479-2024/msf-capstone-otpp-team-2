from flask import Flask, render_template, request, jsonify
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import sys
import os

app = Flask(__name__)

def create_model():
    """Creates and returns the bond optimization model"""
    # Read data
    bond_data = pd.read_csv(r'bond_data.csv')
    
    BenchmarkWeight = 0.01
    
    # Calculate derived fields
    bond_data['transaction_cost'] = bond_data['ask_price'] - bond_data['bid_price']
    bond_data['daily_return'] = bond_data['expected_return']
    bond_data['DTS'] = bond_data['duration'] * bond_data['OAS']
    
    # Model parameters
    N = 100
    lambda_1 = 10
    lambda_2 = 20
    
    # Create model
    model = gp.Model("DynamicCorporateBondOptimization")
    
    # Add variables
    w = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="w")
    x = model.addVars(N, vtype=GRB.BINARY, name="x")
    
    # Set objective
    model.setObjective(
        gp.quicksum((bond_data.loc[i, 'expected_return'] * w[i] - bond_data.loc[i, 'transaction_cost'] * x[i]) for i in range(N))
        - lambda_1 * gp.quicksum(bond_data.loc[i, 'OAS'] * w[i] for i in range(N))
        - lambda_2 * gp.quicksum(bond_data.loc[i, 'DTS'] * w[i] for i in range(N)),
        GRB.MAXIMIZE
    )
    
    # Transaction cost constraints
    Benchmark_cost = gp.quicksum(bond_data.loc[i, 'transaction_cost'] * BenchmarkWeight for i in range(N))
    lower_t_cost = 1 * Benchmark_cost
    upper_t_cost = 1 * Benchmark_cost
    model.addConstr(gp.quicksum(bond_data.loc[i, 'transaction_cost'] * x[i] for i in range(N)) >= lower_t_cost, "MintCost")
    model.addConstr(gp.quicksum(bond_data.loc[i, 'transaction_cost'] * x[i] for i in range(N)) <= upper_t_cost, "MaxtCost")
    
    # # Binary constraints
    # M = 100000
    # for i in range(N):
    #     model.addConstr(w[i] <= M * x[i], f"WeightSelection_{i}")
    
    # Sum of weights constraint
    model.addConstr(gp.quicksum(w[i] for i in range(N)) == 1, "WeightSum")
    
    return model

def approach_1():
    """IIS approach"""
    messages = []
    removed = []
    
    messages.append("Starting IIS analysis...")
    
    try:
        model = create_model()
        
        while True:
            model.computeIIS()
            found_iis = False
            
            for c in model.getConstrs():
                if c.IISConstr:
                    removed.append(str(c.ConstrName))
                    messages.append(f"Found IIS constraint: {c.ConstrName}")
                    model.remove(c)
                    found_iis = True
                    break
            
            if not found_iis:
                messages.append("No more IIS constraints found.")
                break
                    
            model.optimize()
            if model.Status == GRB.OPTIMAL:
                messages.append("Model is now feasible!")
                break
            elif model.Status == GRB.UNBOUNDED:
                messages.append("Warning: Model became unbounded after removing constraints.")
                break
                
        if removed:
            messages.append(f"Summary of removed constraints: {', '.join(removed)}")
    
    except Exception as e:
        messages.append(f"Error in IIS analysis: {str(e)}")
    
    return messages

def approach_2():
    """Constraint relaxation with artificial variables"""
    messages = []
    
    messages.append("Starting constraint relaxation analysis...")
    
    try:
        model = create_model()
        orignumvars = model.NumVars
        
        model.feasRelaxS(0, False, False, True)
        model.optimize()
        
        messages.append("\nAnalyzing relaxation results:")
        slacks = model.getVars()[orignumvars:]
        found_slack = False
        
        for sv in slacks:
            if sv.X > 1e-6:
                found_slack = True
                
                varname = sv.VarName
                if varname.startswith("ArtP_"):
                    constraint_name = varname[5:]  
                    messages.append(f"Constraint {constraint_name}: RHS needs to be decreased by {sv.X:.6f}")
                    
                    # Add specific interpretation for this constraint
                    if constraint_name == "MaxtCost":
                        messages.append(f"   → The maximum transaction cost bound needs to be decreased by {sv.X:.6f}")
                    elif constraint_name == "MintCost":
                        messages.append(f"   → The minimum transaction cost bound needs to be decreased by {sv.X:.6f}")
                    
                elif varname.startswith("ArtN_"):
                    constraint_name = varname[5:]  
                    messages.append(f"Constraint {constraint_name}: RHS needs to be increased by {sv.X:.6f}")
                    
        
                    if constraint_name == "MaxtCost":
                        messages.append(f"   → The maximum transaction cost bound needs to be increased by {sv.X:.6f}")
                    elif constraint_name == "MintCost":
                        messages.append(f"   → The minimum transaction cost bound needs to be increased by {sv.X:.6f}")
                
        if not found_slack:
            messages.append("No constraint relaxations were needed - all constraints can be satisfied.")
        else:
            messages.append(" ")
            
    except Exception as e:
        messages.append(f"Error during relaxation analysis: {str(e)}")
    
    return messages

def approach_3():
    """FeasOpt approach"""
    messages = []
    
    messages.append("Starting FeasOpt analysis...")
    
    try:
        model = create_model()
        model.feasRelaxS(0, True, False, True)
        model.optimize()
        
        if model.Status == GRB.OPTIMAL:
            messages.append("Found optimal relaxation")
            found_relaxation = False
            
            for c in model.getConstrs():
                if abs(c.Slack) > 1e-6:
                    found_relaxation = True
                    messages.append(f"Constraint {c.ConstrName} needed relaxation of {abs(c.Slack):.6f}")
                    
            if not found_relaxation:
                messages.append("No significant constraint relaxations were needed.")
                
            obj_val = model.ObjVal
            messages.append(f"Objective value after relaxation: {obj_val:.4f}")
            
        else:
            messages.append("Could not find feasible relaxation")
            
    except Exception as e:
        messages.append(f"Error during FeasOpt: {str(e)}")
    
    return messages

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    approach = request.form.get('approach')
    
    try:
        if approach == '1':
            messages = approach_1()
        elif approach == '2':
            messages = approach_2()
        elif approach == '3':
            messages = approach_3()
        else:
            return jsonify({'error': 'Invalid approach selected'})
            
        return jsonify({'messages': messages})
        
    except Exception as e:
        return jsonify({'error': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
