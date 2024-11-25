import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import scipy.stats as stats


def mvo_optimize(data, x0):
    #Get the data
    mu = np.array(data["ExpectedReturn"])
    number_of_assets = len(mu)
    OAS = np.array(data["spread"])
    #sectors = np.array(data["sector"])
    duration = np.array(data["ModifiedDuration"])
    liquidity_score = np.array(data["LiquidityScore"])
    std_dev = np.array(data["StdDev"])
    
    #Calculate DTS
    DTS = np.array(data['ModifiedDuration'] *data['spread'])
    
    #Calculate parameters for Robustification (using a box uncertainty set)
    alpha = 0.95
    epsilon = stats.norm.ppf((1+alpha)/2)
    # assume the std devs given represent the standard error from estimation of the mean
    # basically assuming that root(T) = 1
    delta = std_dev*epsilon
    #Define Sectors and Sector Weights
    #Prep the sector data for use
    # sector_unique = np.unique(sectors).tolist()
    # sector_weights = {}
    # for sector in sector_unique:
    # #!!!!IN THE ACTUAL INTERFACE YOU NEED TO ASK USER WHAT THESE WEIGHTS ARE!!!!
    # #Set to -1 if user doesn't care
    #     sector_weights[sector] = .05
        

    #Define benchmark weights
    benchmark = np.ones(number_of_assets)/number_of_assets
    
    #Define parameters, in the future this will be inputted
    #if the user chooses not to add a constraint, then set to -1
    robust = True
    dev_bench = -1
    dev_bench_OAS = np.mean(OAS) 
    target_return = np.mean(mu)
    change_in_weight = -1 # x0 defined as function input
    dev_bench_duration = np.mean(duration) 
    min_trade_size = -1 #not currently live
    max_credit_risk = np.mean(DTS)
    liquidity = np.mean(liquidity_score) 
    diversity = 0.4
    
    constraint_values= {
    "Deviation from Benchmark" : dev_bench,
    "Deviation from Benchmark OAS": dev_bench_OAS,
    "Target Return" : target_return,
    "Change in Portfolio Weight" : change_in_weight,
    "Deviation from Benchmark Duration": dev_bench_duration,
    "Sector Weights" : 0, #sector_weights,
    "Maximum Credit Risk": max_credit_risk, 
    "Liquidity" : liquidity,
    "Minimum Trade Size": min_trade_size,
    "Benchmark OAS": np.mean(OAS),
    "Benchmark Duration": np.mean(duration),   
    "Diversity": diversity
    }


    # ADD CONSTRAINTS TO MODEL
    #Define model
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    
    #define weights with postivity constraint
    weights = model.addVars(number_of_assets, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="Weight")
    #auxillary vars represeting the abs value of OAS/trade size
    oas_aux = model.addVar( vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Auxillary Var OAS")
    duration_aux = model.addVar( vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Auxillary Var Duration")
    trade_aux = model.addVars(number_of_assets, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Auxillary Var Trade Size")
    weight_change_aux = model.addVars(number_of_assets, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="Auxillary Var Weight Change")
    
    #ADD CONSTRAINTS
    
    #all wealth invested
    model.addConstr(sum(weights[i] for i in range(number_of_assets)) == 1, name = "Wealth Constraint")
    
    #### deviation from benchmark
    
    if constraint_values["Deviation from Benchmark"] > -1:
        for i in range(number_of_assets):
            #implement absolute value constraints
            model.addConstr(weights[i]-benchmark[i] <= constraint_values["Deviation from Benchmark"])
            model.addConstr(weights[i]-benchmark[i] >= -1*constraint_values["Deviation from Benchmark"])
    
    #Deviation from Benchmark OAS
    if constraint_values["Deviation from Benchmark OAS"] > -1:
        #oas_aux represents the absolute value of OAS benchmark
        model.addConstr(sum(weights[i]*OAS[i] for i in range(number_of_assets)) -constraint_values["Benchmark OAS"] <= oas_aux)
        model.addConstr(sum(weights[i]*OAS[i] for i in range(number_of_assets)) -constraint_values["Benchmark OAS"] >= -1*oas_aux)
        model.addConstr(oas_aux <= constraint_values["Deviation from Benchmark OAS"], name = "OAS Benchmark")
    
    #Target return
        # delta here is the robust term
    if constraint_values["Target Return"] > -1:
        if robust:
            model.addConstr(sum(weights[i]*(mu[i] - delta[i])  for i in range(number_of_assets)) >= constraint_values["Target Return"])
        else:
            model.addConstr(sum(weights[i]*(mu[i])  for i in range(number_of_assets)) >= constraint_values["Target Return"])
        
    #Change in Portfolio Weight
    if constraint_values["Change in Portfolio Weight"] >-1:
        for i in range(number_of_assets):
            model.addConstr(weights[i]-x0[i] <= weight_change_aux[i])
            model.addConstr(weights[i]-x0[i] >= -1*weight_change_aux[i])
        model.addConstr(gp.quicksum(weight_change_aux) <= constraint_values["Change in Portfolio Weight"], name = "Change in Weights")
    
    #Deviation from Benchmark Duration
    if constraint_values["Deviation from Benchmark Duration"] > -1:
        model.addConstr(sum(weights[i]*duration[i] for i in range(number_of_assets)) -constraint_values["Benchmark Duration"] <= duration_aux)
        model.addConstr(sum(weights[i]*duration[i] for i in range(number_of_assets)) -constraint_values["Benchmark Duration"] >= -1*duration_aux)
        model.addConstr(duration_aux <= constraint_values["Deviation from Benchmark Duration"], name = "Duration Benchmark")
    
    #MIN TRADE SIZE
    # if min_trade_size >-1:
    #     for i in range(number_of_assets):
    #         model.addConstr((weights[i]-x0[i])*OAS[i] <= trade_aux[i])
    #         model.addConstr((weights[i]-x0[i])*OAS[i] >= -1*trade_aux[i])
    #         model.addConstr(gp.quicksum(trade_aux) <= min_trade_size, name = "Trade Size Min")
    
    #SECTOR
    # for i in range(number_of_assets):
    #     if constraint_values["Sector Weights"][sectors[i]] >-1:
    #         model.addConstr(weights[i] <= constraint_values["Sector Weights"][sectors[i]])
            
    # DTS
    if constraint_values["Maximum Credit Risk"] > -1:
        model.addConstr(sum(weights[i]*DTS[i] for i in range(number_of_assets)) <= max_credit_risk)
        
    # LIQUIDITY SCORE
    if constraint_values["Liquidity"]>-1:
        model.addConstr(sum(weights[i]*liquidity_score[i] for i in range(number_of_assets)) >= constraint_values["Liquidity"])
        
    #Diversity Constraint
    if constraint_values["Diversity"] >-1:
        for i in range(number_of_assets):
            model.addConstr(weights[i] <= constraint_values["Diversity"])
        
    # Set Objective
    if robust:
        model.setObjective( sum(-1*weights[i]*(mu[i] - delta[i]) for i in range(number_of_assets)) + sum(weights[i]*DTS[i] for i in range(number_of_assets)), gp.GRB.MINIMIZE)
    else:
        model.setObjective( sum(-1*weights[i]*mu[i] for i in range(number_of_assets)) + sum(weights[i]*DTS[i] for i in range(number_of_assets)), gp.GRB.MINIMIZE)
        
        
    #optimize
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        print("\n", "infeasbile")
        model.computeIIS()
        model.write("model.ilp")
    if model.status == GRB.UNBOUNDED:
        print("The model is unbounded.")
    return np.array([weights[i].X for i in range(number_of_assets)])