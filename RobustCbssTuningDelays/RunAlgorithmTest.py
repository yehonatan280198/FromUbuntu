import math
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
gurobiModel.setParam("TimeLimit", 20)
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
delay_prob_Exec = float(sys.argv[4])
algorithm = sys.argv[5]
configStr = f"{algorithm}_{mapName}_num_of_agents_{num_of_agents}_num_of_goals_{num_of_goals}_delay_prob_Exec_{delay_prob_Exec}"
if algorithm == "Baselines":
    desired_safe_probs_for_test = ["NotAvailable", 0]
    algorithm = "Strict"
else:
    desired_safe_probs_for_test = [0.05, 0.25, 0.5, 0.8, 0.95, 0.99, 0.999, 0.9999]

instances = 75
verifyAlpha = 0.05
max_planning_time = 60

####################################################### Write the header of a CSV file ############################################################
if not os.path.exists("Output_files"):
    os.makedirs("Output_files")

columns = ["Map", "Desired Safe prob", "Delay prob (Planning)", "Delay prob (Execution)", "Number of agents",
           "Number of goals", "Instance", "Runtime", "Offline Runtime", "Online Runtime", "Number Of Replans",
           "Online Sum of Service Time", "Offline Sum of Service Time", "Number of Expands", "Min Safe Prob"]

with open(f"Output_files/Output_{configStr}.csv", mode="w", newline="",
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
def run_Test(desired_safe_prob, AgentLocations, GoalLocations, DelaysProbDictPlanning, DelaysProbDictExecution):
    reset_gurobi_model(gurobiModel)
    randGen = random.Random(44)
    minSafeProb = math.inf
    start_time = time.time()

    # Offline stage
    p, OfflineTime, countExpand = run_robust_planner_with_timeout(AgentLocations, GoalLocations, desired_safe_prob,
                                                                  DelaysProbDictPlanning, mapAndDim, verifyAlpha,
                                                                  gurobiModel,
                                                                  max_planning_time, algorithm, "SST")
    if p is None:
        return None, None, None, None, None, countExpand, minSafeProb

    minSafeProb = min(minSafeProb, round(p[2], 3)) if desired_safe_prob != "NotAvailable" else "NotAvailable"
    Offline_SST = p[1]
    OnlineTime, numOfReplans, timestep, Online_SST = 0, 0, 0, 0

    while True:
        if time.time() - start_time >= 300:
            return (round(OfflineTime, 3), None, numOfReplans, None, Offline_SST,
                    round(countExpand / (numOfReplans + 1), 3), minSafeProb)

        s = Run_Simulation(p[0], DelaysProbDictExecution, AgentLocations, GoalLocations, randGen, timestep, Online_SST)
        if s.runSimulation():
            Online_SST = s.TST
            break

        AgentLocations, GoalLocations = s.AgentLocations, s.remainGoals

        if s.TST - Online_SST != 0:
            reset_gurobi_model(gurobiModel)

        # Online re-planning
        p, replan_time, currCountExpand = run_robust_planner_with_timeout(AgentLocations, GoalLocations,
                                                                          desired_safe_prob,
                                                                          DelaysProbDictPlanning, mapAndDim,
                                                                          verifyAlpha, gurobiModel,
                                                                          max_planning_time, algorithm, "SST")

        countExpand += currCountExpand
        if p is None:
            return (round(OfflineTime, 3), None, numOfReplans + 1, None, Offline_SST,
                    round(countExpand / (numOfReplans + 2), 3), minSafeProb)

        minSafeProb = min(minSafeProb, round(p[2], 4)) if desired_safe_prob != "NotAvailable" else "NotAvailable"
        OnlineTime += replan_time
        numOfReplans += 1
        timestep = s.timestep
        Online_SST = s.TST

    return (round(OfflineTime, 3), round(OnlineTime, 3), numOfReplans, Online_SST, Offline_SST,
            round(countExpand / (numOfReplans + 1), 3), minSafeProb)


####################################################### run Tests #################################################################################

def run_instances():
    delaysProbDictForExecution = {i: delay_prob_Exec for i in range(num_of_agents)}

    for instance in range(1, instances + 1):
        temp_records = []

        for curr_desired_safe_prob in desired_safe_probs_for_test:
            delay_prob_plan = delay_prob_Exec if curr_desired_safe_prob != "NotAvailable" else 0

            delaysProbDictForPlanning = {i: delay_prob_plan for i in range(num_of_agents)}
            AgentsLocations, GoalsLocations = read_locs_from_file(instance)

            print(
                f"\nmap: {mapName}, desired safe prob: {curr_desired_safe_prob}, planning delay prob: {delay_prob_plan}, "
                f"execution delay prob: {delay_prob_Exec}, agents: {len(AgentsLocations)}, goals: {len(GoalsLocations)}, instance: {instance}")

            result = run_Test(curr_desired_safe_prob, AgentsLocations, GoalsLocations, delaysProbDictForPlanning,
                              delaysProbDictForExecution)
            offlineRuntime, onlineRuntime, numOfReplans, sstOnline, sstOffline, CountExpand, MinSafeProb = result

            if onlineRuntime is None:
                print("Plan is None!\n--------------------------------------------------------------------------\n")
                temp_records.append([mapName, curr_desired_safe_prob, delay_prob_plan, delay_prob_Exec, num_of_agents,
                                     num_of_goals, instance, None, offlineRuntime, onlineRuntime, numOfReplans,
                                     sstOnline, sstOffline, CountExpand, MinSafeProb])
                continue

            runtime = round(offlineRuntime + onlineRuntime, 3)
            print("Pass!\n--------------------------------------------------------------------------\n")

            temp_records.append(
                [mapName, curr_desired_safe_prob, delay_prob_plan, delay_prob_Exec, num_of_agents, num_of_goals,
                 instance, runtime, offlineRuntime, onlineRuntime, numOfReplans, sstOnline, sstOffline, CountExpand, MinSafeProb])

        print(f"All safe_prob runs succeeded for instance {instance}, writing to CSV...\n")
        with open(f"Output_files/Output_{configStr}.csv", mode="a", newline="", encoding="utf-8") as file:
            writerRecord = csv.writer(file)
            writerRecord.writerows(temp_records)


run_instances()
