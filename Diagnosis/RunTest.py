import os
import random
import csv
import ast
import sys

from Robust_Planner import run_robust_planner_with_timeout
from Run_Simulation import Run_Simulation
import gurobipy as gp


####################################################### Create Map #################################################################################
def create_map(map_name):
    file_path = f"OurResearch.domain/{map_name}.map"
    with open(file_path, "r") as file:
        lines = file.readlines()
    map_start_index = lines.index("map\n") + 1
    map_lines = lines[map_start_index:]

    currMap = []
    rows, cols = 0, 0
    for line in map_lines:
        cols = len(line.strip())
        rows += 1
        for char in line.strip():
            currMap += [0] if char == "." else [1]

    return {"Rows": rows, "Cols": cols, "Map": currMap}


####################################################### Global Variables ######################################################################
mapName = sys.argv[1]
mapAndDim = create_map(mapName)
num_of_agents, num_of_goals = int(sys.argv[2]), int(sys.argv[3])
obstacles_agents = True if sys.argv[4] == "True" else False
type_of_allocation = sys.argv[5]
configStr = (f"{mapName}-{num_of_agents}_agents-{num_of_goals}_goals-{obstacles_agents}"
             f"_obstacles_agents-{type_of_allocation}_allocation")

instances = 300
verifyAlpha = 0.05
max_planning_time = 60
p_delays_list = [0.1, 0.3, 0.5, 0.7]
desired_safe_prob = 0.6


####################################################### Gurobi model #################################################################################
def reset_gurobi_model(model):
    model.update()
    for constr in model.getConstrs():
        model.remove(constr)

    for var in model.getVars():
        model.remove(var)

    model.setObjective(0)
    model.update()


gurobiModel = None
if type_of_allocation == "Gurobi":
    gurobiModel = gp.Model("MinimizeTotalServiceTime")
    gurobiModel.setParam("OutputFlag", 0)
    gurobiModel.setParam("TimeLimit", 20)
    gurobiModel.setParam("IntFeasTol", 1e-9)
    gurobiModel.setParam("Seed", 42)
    gurobiModel.setParam("Threads", 1)
    gurobiModel.setParam("MIPFocus", 1)
    gurobiModel.setParam("Heuristics", 0.8)
    gurobiModel.setParam("ImproveStartTime", 0.0)

####################################################### Write the header of a CSV file ############################################################
if not os.path.exists("Output_files"):
    os.makedirs("Output_files")

columns = ["Map", "Inactive agent acts as an obstacle", "Desired Safe prob", "Number of agents", "Number of goals",
           "Type of allocation", "Instance", "Runtime", "Offline Runtime", "Online Runtime",
           "Number Of Replans", "Online SST", "Sum of Planning Times for All Goals", "SGAT",
           "Offline SST", "Number of Expands"]

with open(f"Output_files/Output_{configStr}.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

####################################################### Read locs from file #################################################################################
def read_locs_from_file(num_of_instance):
    file_agents_name = f"Agent_Goal_locations_files/{mapName.split('.')[0]}_Map_Agent_Locs_instance_{num_of_instance - 1}.txt"
    file_goals_name = f"Agent_Goal_locations_files/{mapName.split('.')[0]}_Map_Goal_Locs_instance_{num_of_instance - 1}.txt"

    with open(file_agents_name, "r") as f:
        Agents_Positions = [ast.literal_eval(line.strip()) for _, line in zip(range(num_of_agents), f)]

    with open(file_goals_name, "r") as f:
        Goals_Locations = [ast.literal_eval(line.strip()) for _, line in zip(range(num_of_goals), f)]

    return Agents_Positions, Goals_Locations

def run_Test(AgentLocations, GoalLocations, DelaysProbDictExecution):
    if type_of_allocation == "Gurobi":
        reset_gurobi_model(gurobiModel)
    randGen = random.Random(44)

    p, OfflineTime, countExpand = run_robust_planner_with_timeout(AgentLocations, GoalLocations, desired_safe_prob,
                                                                  DelaysProbDictExecution, mapAndDim, verifyAlpha,
                                                                  gurobiModel, max_planning_time,
                                                                  obstacles_agents, type_of_allocation)
    if p is None:
        return None, None, None, None, None, None, countExpand

    plan_paths, Offline_SST, inactive_agents = p
    planning_time_sum_over_goals = OfflineTime * len(GoalLocations)

    OnlineTime, numOfReplans, timestep, Online_SST = 0, 0, 0, 0

    s = Run_Simulation(plan_paths, DelaysProbDictExecution, AgentLocations, GoalLocations, randGen, timestep,
                       Online_SST, inactive_agents, obstacles_agents)

    if s.runSimulation():
        return round(OfflineTime, 3), round(OnlineTime, 3), 0, s.TST, Offline_SST, round(
            planning_time_sum_over_goals, 3), round(countExpand / (numOfReplans + 1), 3)

    return round(OfflineTime, 3), None, 1, None, Offline_SST, None, round(
        countExpand / (numOfReplans + 1), 3)

####################################################### run Tests #################################################################################

def run_instances():
    randGenForDelays = random.Random(46)

    for instance in range(1, instances + 1):
        temp_records = []
        delaysProbDictForExecution = {i: randGenForDelays.choice(p_delays_list) for i in range(num_of_agents)}

        AgentsLocations, GoalsLocations = read_locs_from_file(instance)

        print(f"\nmap: {mapName}, Inactive agent acts as an obstacle: {obstacles_agents}, agents: {num_of_agents}, "
              f"goals: {num_of_goals}, instance: {instance}")

        # print(f"Agents Positions: {AgentsLocations} \nGoals Locations: {GoalsLocations} \nAgents Delays: {delaysProbDictForExecution}")

        result = run_Test(AgentsLocations, GoalsLocations, delaysProbDictForExecution)
        offlineRuntime, onlineRuntime, numOfReplans, sstOnline, sstOffline, Planning_time_sum_over_goals, CountExpand = result

        if onlineRuntime is None:
            print("Plan is None!\n--------------------------------------------------------------------------\n")
            temp_records.append([
                mapName, obstacles_agents, desired_safe_prob, num_of_agents, num_of_goals, type_of_allocation,
                instance, None, offlineRuntime, onlineRuntime, numOfReplans, sstOnline, Planning_time_sum_over_goals,
                None, sstOffline, CountExpand
            ])

        else:
            runtime = round(offlineRuntime + onlineRuntime, 3)
            SGAT = sstOnline + Planning_time_sum_over_goals
            print("Pass!\n--------------------------------------------------------------------------\n")

            temp_records.append([
                mapName, obstacles_agents, desired_safe_prob, num_of_agents, num_of_goals, type_of_allocation,
                instance, runtime, offlineRuntime, onlineRuntime, numOfReplans, sstOnline, Planning_time_sum_over_goals,
                SGAT, sstOffline, CountExpand
            ])

        print(f"instance {instance} writing to CSV...")
        with open(f"Output_files/Output_{configStr}.csv", mode="a", newline="", encoding="utf-8") as file:
            writerRecord = csv.writer(file)
            writerRecord.writerows(temp_records)


run_instances()