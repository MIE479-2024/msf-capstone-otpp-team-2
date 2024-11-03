import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

def define_model(constraint_values):
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
        model.addConstr(sum(weights[i]*mu[i] for i in range(number_of_assets)) >= constraint_values["Target Return"])
        
    #Change in Portfolio Weight
    #Need to define how we are doing this? Depends on the iteration method
    
    #x0 not inputted yet
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
        
    model.setObjective(sum(weights[i]*standard_deviation[i]*standard_deviation[i] for i in range(number_of_assets)) - sum(weights[i]*mu[i] for i in range(number_of_assets)) + sum(weights[i]*DTS[i] for i in range(number_of_assets)), gp.GRB.MINIMIZE)
    return model

#Now apply the above function to the actual data

#Get data
def get_column(file_path, column_name):
    df = pd.read_csv(file_path)
    if column_name in df.columns:
        return df[column_name].tolist()
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the Excel file.")

filepath = "bond_data_rev1.csv"

#Define vars that will be user inputs later

#Q = np.array([[1, 2], [3, 4]]) -not using Q atm
standard_deviation = get_column(filepath, "return_std_dev")
mu = get_column(filepath, "expected_return")
number_of_assets = len(mu)
OAS = get_column(filepath, "OAS")
sectors = get_column(filepath, "sector")
duration = get_column(filepath, "duration")
liquidity_score = get_column(filepath, "liquidity_score")

## X0 Not defined correctly yet ##
x0 = [1/number_of_assets]*number_of_assets

## NEED TO DEFINE BENCHMARK LATER ##
benchmark = [.2, .8]

#Prep the sector data for use
sector_unique = list(set(sectors))
sector_weights = {}
for sector in sector_unique:
    #!!!!IN THE ACTUAL INTERFACE YOU NEED TO ASK USER WHAT THESE WEIGHTS ARE!!!!
    #Set to -1 if user doesn't care
    sector_weights[sector] = 0.5

#Calculate the DTS
DTS = []
for i in range(number_of_assets):
    DTS.append(duration[i] * OAS[i])

#Define parameters, in the future this will be inputted
    #if the user chooses not to add a constraint, then set to -1 (should be ok cuz all params nonneg)
dev_bench = -1
dev_bench_OAS = -1
target_return = -1
change_in_weight = -1
dev_bench_duration = -1
min_trade_size = -1
max_credit_risk = -1
liquidity = -1

    

user_inputted_constraints = {
    "Deviation from Benchmark" : dev_bench,
    "Deviation from Benchmark OAS": dev_bench_OAS,
    "Target Return" : target_return,
    "Change in Portfolio Weight" : change_in_weight,
    "Deviation from Benchmark Duration": dev_bench_duration,
    "Sector Weights" : sector_weights,
    "Maximum Credit Risk": max_credit_risk, 
    "Liquidity" : liquidity,
    "Minimum Trade Size": min_trade_size
    }

    

# ADD CONSTRAINTS TO MODEL
model1 = define_model(user_inputted_constraints)

model1.setParam('OutputFlag', 0)
model1.optimize()

#Now need to deal with possible infeasibility


def not_optimal(user_inputted_constraints):
    #null constraints -- relaxed to the point of being obsolete
    null_constraints ={
        "Deviation from Benchmark": gp.GRB.INFINITY,
        "Deviation from Benchmark OAS": gp.GRB.INFINITY,
        "Target Return": -gp.GRB.INFINITY,
        "Change in Portfolio Weight": gp.GRB.INFINITY,
        "Deviation from Benchmark Duration":gp.GRB.INFINITY,
        "Sector Weights": gp.GRB.INFINITY,
        "Liquidity": -gp.GRB.INFINITY,
        "Maximum Credit Risk":gp.GRB.INFINITY,
        "Minimum Trade Size": -gp.GRB.INFINITY
    }
    found_optimal = False
    #Now lets iterate through relaxed constraints and replace them with the null
    for key in user_inputted_constraints:
        new_dictionary = user_inputted_constraints
        if user_inputted_constraints[key]!= -1:
            new_dictionary[key] = null_constraints[key]
        model_null = define_model(new_dictionary)
        model_null.setParam('OutputFlag', 0)
        model_null.optimize()
        if model_null.status == gp.GRB.OPTIMAL:
            print("Removing the following constraint resulted in an optimal model:", key)
            found_optimal = True
            
    #ok lets do pairs now
    if not found_optimal:
        for key1 in user_inputted_constraints:
            for key2 in user_inputted_constraints:
                if user_inputted_constraints[key1] != -1 and user_inputted_constraints[key2] != -1 and key1 != key2:
                    print("Trying this pair:", key1, key2)
                    new_dictionary = user_inputted_constraints
                    new_dictionary[key1] = null_constraints[key1]
                    new_dictionary[key2] = null_constraints[key2]
                    model_pair = define_model(new_dictionary)
                    model_pair.setParam('OutputFlag', 0)
                    model_pair.optimize
                    if model_pair.status == gp.GRB.OPTIMAL:
                        print("Removing the Following Two constraitnts resulted in an optimal model:", key1, key2)
        
if model1.status != gp.GRB.OPTIMAL:
    print("Model is infeasible. Relax Constraints.")
    not_optimal(user_inputted_constraints)
    
#Use DTS -- forward predictor of volatitliy -- no correlation -- sector correlations (aggregations) -- use OAS as a measure of risk