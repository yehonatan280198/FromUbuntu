import os
import random
import csv
import ast
import sys
import time

from Robust_Planner import run_robust_planner_with_timeout
from Run_Simulation import Run_Simulation
import gurobipy as gp


def reset_gurobi_model(model):
    model.update()
    for constr in model.getConstrs():
        model.remove(constr)

    for var in model.getVars():
        model.remove(var)

    model.setObjective(0)
    model.update()


gurobiModel = gp.Model("MinimizeTotalServiceTime")
gurobiModel.setParam("OutputFlag", 0)
gurobiModel.setParam("IntFeasTol", 1e-9)
gurobiModel.setParam("Seed", 42)


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
num_of_agents = int(sys.argv[2])
num_of_goals = int(sys.argv[3])
optimize = sys.argv[4]
configStr = f"{mapName}_num_of_agents_{num_of_agents}_num_of_goals_{num_of_goals}_optimize_{optimize}"

instances = 50
max_planning_time = 300

####################################################### Write the header of a CSV file ############################################################
if not os.path.exists("Type_Of_Optimize_Test_files"):
    os.makedirs("Type_Of_Optimize_Test_files")

columns = ["Map", "Number of agents", "Number of goals", "Instance", "Optimize", "Runtime", "Online Sum of Service Time",
           "Number of Expands"]

with open(f"Type_Of_Optimize_Test_files/Output_{configStr}.csv", mode="w", newline="",
          encoding="utf-8") as file:
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


####################################################### run Test  #################################################################################
def run_Test(AgentLocations, GoalLocations, DelaysProbDictExecution):
    reset_gurobi_model(gurobiModel)
    randGen = random.Random(44)
    start_time = time.time()

    # Offline stage
    p, OfflineTime, countExpand = run_robust_planner_with_timeout(AgentLocations, GoalLocations, "NotAvailable",
                                                                  DelaysProbDictExecution, mapAndDim, 0.05, gurobiModel,
                                                                  max_planning_time, "Strict", optimize)
    if p is None:
        return None, None, countExpand

    OnlineTime, numOfReplans, timestep, Online_TST = 0, 0, 0, 0

    while True:
        if time.time() - start_time >= 1000:
            return round(OfflineTime, 3), None, round(countExpand / (numOfReplans + 1), 3)

        s = Run_Simulation(p[0], DelaysProbDictExecution, AgentLocations, GoalLocations, randGen, timestep, Online_TST)
        if s.runSimulation():
            Online_TST = s.TST
            break

        AgentLocations, GoalLocations = s.AgentLocations, s.remainGoals

        if s.TST - Online_TST != 0:
            reset_gurobi_model(gurobiModel)

        # Online re-planning
        p, replan_time, currCountExpand = run_robust_planner_with_timeout(AgentLocations, GoalLocations,
                                                                          "NotAvailable",
                                                                          DelaysProbDictExecution, mapAndDim,
                                                                          0.05, gurobiModel,
                                                                          max_planning_time, "Strict", optimize)

        countExpand += currCountExpand
        if p is None:
            return round(OfflineTime, 3), None, round(countExpand / (numOfReplans + 2), 3)

        OnlineTime += replan_time
        numOfReplans += 1
        timestep = s.timestep
        Online_TST = s.TST

    return round(OfflineTime + OnlineTime, 3), Online_TST, round(countExpand / (numOfReplans + 1), 3)

####################################################### run Tests #################################################################################

def run_instances():
    delaysProbDictForExecution = {i: 0 for i in range(num_of_agents)}
    for instance in range(1, instances + 1):
        AgentsLocations, GoalsLocations = read_locs_from_file(instance)

        print(f"map: {mapName}, agents: {AgentsLocations}, goals: {GoalsLocations}, optimize: {optimize}")

        result = run_Test(AgentsLocations, GoalsLocations, delaysProbDictForExecution)
        runtime, sstOnline, countExpand = result
        record = [mapName, num_of_agents, num_of_goals, instance, optimize, runtime, sstOnline, countExpand]

        print(f"Instance {instance} is writing to CSV...\n")
        with open(f"Type_Of_Optimize_Test_files/Output_{configStr}.csv", mode="a", newline="", encoding="utf-8") as file:
            writerRecord = csv.writer(file)
            writerRecord.writerow(record)


run_instances()
