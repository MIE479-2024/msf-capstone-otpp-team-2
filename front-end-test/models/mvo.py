import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

def mvo_optimize(bond_data, x0, constraints):
    
    print(constraints)
 


    #Get the data
    mu = np.array(bond_data["expected_return"])
    number_of_assets = len(mu)
    OAS = np.array(bond_data["OAS"])
    sectors = bond_data["sector"] 
    duration = np.array(bond_data["duration"])
    liquidity_score = np.array(bond_data["liquidity_score"])
    
    #Calculate DTS
    bond_data["DTS"] = bond_data['duration'] * bond_data['OAS']
    DTS = np.array(bond_data["DTS"])

    #Define Sectors and Sector Weights -- Skip for Now bc causing issues
    #Prep the sector data for use
    sector_unique = list(set(sectors))
    sector_weights = {}
    for sector in sector_unique:
    #!!!!IN THE ACTUAL INTERFACE YOU NEED TO ASK USER WHAT THESE WEIGHTS ARE!!!!
    #Set to -1 if user doesn't care
        sector_weights[sector] = -1
        

    ## NEED TO DEFINE BENCHMARK LATER ##
    #benchmark = 
    
    #Define parameters, in the future this will be inputted
    #if the user chooses not to add a constraint, then set to -1 (should be ok cuz all params nonneg)
    dev_bench = -1
    dev_bench_OAS = -1
    target_return = 8
    change_in_weight = .2
    dev_bench_duration = -1
    min_trade_size = -1
    max_credit_risk = -1 #not working
    liquidity = 3
     #Define model
    model = gp.Model()

    
    #define weights with postivity constraint
    weights = model.addVars(number_of_assets, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Weight")
    #auxillary vars represeting the abs value of OAS/trade size
    oas_aux = model.addVars(number_of_assets, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Auxillary Var OAS")
    duration_aux = model.addVars(number_of_assets, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Auxillary Var Duration")
    trade_aux = model.addVars(number_of_assets, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Auxillary Var Trade Size")
    weight_change_aux = model.addVars(number_of_assets, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Auxillary Var Weight Change")
    
    #all wealth invested
    model.addConstr(sum(weights[i] for i in range(number_of_assets)) == 1, name = "Wealth Constraint")
    
    #### deviation from benchmark and Benchmark OAS ####
    # CURRENTLY BENCHMARK IS NOT WELL DEFINED!!!
    # if constraint_values["Deviation from Benchmark"] > -1:
    #     for i in range(number_of_assets):
    #         #implement absolute value constraints
    #         model.addConstr(weights[i]-benchmark[i] <= constraint_values["Deviation from Benchmark"])
    #         model.addConstr(weights[i]-benchmark[i] >= -1*constraint_values["Deviation from Benchmark"])
    # if constraint_values["Deviation from Benchmark OAS"] > -1:
    #     #oas_aux represents the absolute value of OAS benchmark
    #     model.addConstr((weights[i]-benchmark[i])*OAS[i] <= oas_aux[i])
    #     model.addConstr((weights[i]-benchmark[i])*OAS[i] >= -1*oas_aux[i])
    #     model.addConstr(gp.quicksum(oas_aux) <= constraint_values["Deviation from Benchmark OAS"], name = "OAS Benchmark")
    
    #Target return
    if target_return > -1:
        model.addConstr(sum(weights[i]*mu[i] for i in range(number_of_assets)) >= target_return)
        
    #Change in Portfolio Weight
    
    # if constraint_values["Change in Portfolio Weight"] >-1:
    #     model.addConstr(weights[i]-x0[i] <= weight_change_aux[i])
    #     model.addConstr(weights[i]-x0[i] >= -1*weight_change_aux[i])
    #     model.addConstr(gp.quicksum(weight_change_aux) <= constraint_values["Change in Portfolio Weight"], name = "Change in Weights")
    
    #DURATION - NEEDS BENCHMARK
    # if constraint_values["Deviation from Benchmark Duration"] > -1:
    #     for i in range(number_of_assets):
    #         model.addConstr((weights[i]-benchmark[i])*duration[i] <= duration_aux[i])
    #         model.addConstr((weights[i]-benchmark[i])*duration[i] >= -1*duration_aux[i])
    #         model.addConstr(gp.quicksum(duration_aux) <= constraint_values["Deviation from Benchmark Duration"], name = "Duration Benchmark")
    
    #MIN TRADE SIZE
    # if min_trade_size >-1:
    #     for i in range(number_of_assets):
    #         model.addConstr((weights[i]-x0[i])*OAS[i] <= trade_aux[i])
    #         model.addConstr((weights[i]-x0[i])*OAS[i] >= -1*trade_aux[i])
    #         model.addConstr(gp.quicksum(trade_aux) <= min_trade_size, name = "Trade Size Min")
    #SECTOR
    for i in range(number_of_assets):
        if sector_weights[sectors[i]] >-1:
            model.addConstr(weights[i] <= sector_weights[sectors[i]])
            
    # DTS - issues with the definition of market yield here
    if max_credit_risk > -1:
        model.addConstr(sum(weights[i]*DTS[i] for i in range(number_of_assets)) <= max_credit_risk)
        
    # LIQUIDITY SCORE
    if liquidity >-1:
        model.addConstr(sum(weights[i]*liquidity_score[i] for i in range(number_of_assets)) >= liquidity)
    
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

    model.setObjective( sum(-1*weights[i]*mu[i] for i in range(number_of_assets)) + sum(weights[i]*DTS[i] for i in range(number_of_assets)), gp.GRB.MINIMIZE)
    model.setParam('OutputFlag', 0)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        return [weights[i].X for i in range(number_of_assets)]
    else:
        print("No optimal solution found.")
        return None
