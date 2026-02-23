import math
import os
import random
import csv
import ast
import sys
import time

from GraphG import GraphG
from Robust_Planner import run_robust_planner_with_timeout
from Run_Simulation import Run_Simulation

utilityFuncDict = {"distanceUtilityWithDelay": (1, 0, 0),
                   "distanceUtilityWithoutDelay": (1, 0, 0)}

seedPerInstanceDict = {"1_10": 101, "11_20": 102, "21_30": 103, "31_40": 104, "41_50": 105,
                       "51_60": 106, "61_70": 107, "71_80": 108, "81_90": 109, "91_100": 110,
                       "101_110": 111, "111_120": 112, "121_130": 113, "131_140": 114, "141_150": 115,
                       "151_160": 116, "161_170": 117, "171_180": 118, "181_190": 119, "191_200": 120}


def getExperiments(exp):
    if exp not in utilityFuncDict:
        return exp, None
    else:
        return exp, utilityFuncDict[exp]


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
experiment, parameters = getExperiments(sys.argv[4])
instanceRange = list(map(int, sys.argv[5].split("_")))
seedPerInstanceForDelay = seedPerInstanceDict[sys.argv[5]]
obstacleAgents = True if sys.argv[6] == "True" else False
configStr = f"M={mapName}--A={num_of_agents}--G={num_of_goals}--E={experiment}--I={sys.argv[5]}--O={obstacleAgents}"

verifyAlpha = 0.05
max_planning_time = 60
desired_safe_prob = 0.7
max_instance_time = 300

####################################################### Write the header of a CSV file ############################################################
if not os.path.exists("Output_files"):
    os.makedirs("Output_files")

columns = ["Map", "Desired Safe prob", "Number of agents", "Number of goals", "Instance", "Experiment",
           "Obstacle Agents", "Offline Runtime", "Online Runtime", "Runtime", "SGAT",
           "Num Of Replan", "Online SST", "Offline SST", "Number of Expands", "Min Safe Prob"]

with open(f"Output_files/Output_{configStr}.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()


####################################################### Read locs from file #################################################################################
def read_locs_from_file(num_of_instance):
    file_agents_name = f"Agent_Goal_locations_files/Agents_Scale/{mapName.split('.')[0]}_Map_Agent_Locs_instance_{num_of_instance - 1}.txt"
    file_goals_name = f"Agent_Goal_locations_files/Agents_Scale/{mapName.split('.')[0]}_Map_Goal_Locs_instance_{num_of_instance - 1}.txt"

    with open(file_agents_name, "r") as f:
        Agents_Positions = [ast.literal_eval(line.strip()) for _, line in zip(range(num_of_agents), f)]

    with open(file_goals_name, "r") as f:
        Goals_Locations = [ast.literal_eval(line.strip()) for _, line in zip(range(num_of_goals), f)]

    return Agents_Positions, Goals_Locations


def run_Test(AgentLocations, GoalLocations, DelaysProbDictExecution, Experiment, Agents_to_remove, Parameters):
    randGen = random.Random(44)
    start_time = time.time()
    minSafeProb = math.inf

    ####################################### Offline Planning #############################################
    print("Calc Plan")
    p, OfflineTime, countExpand = run_robust_planner_with_timeout(
        AgentLocations, GoalLocations, desired_safe_prob, DelaysProbDictExecution, mapAndDim, verifyAlpha,
        max_planning_time, obstacleAgents, Experiment, Agents_to_remove, Parameters)

    if p is None:
        print("Plan is None!\n--------------------------------------------------------------------------")
        return 60.0, None, None, None, None, None, countExpand, None, None

    plan_paths, Offline_SST, currSafeProb, inactive_agents = p
    minSafeProb = min(minSafeProb, round(currSafeProb, 4))
    OnlineTime, numOfReplans, timestep, Online_SST = 0, 0, 0, 0
    SGAT = len(GoalLocations) * OfflineTime

    while True:
        if time.time() - start_time >= max_instance_time:
            return (round(OfflineTime, 3), None, 300.0, numOfReplans, None, Offline_SST,
                    round(countExpand / (numOfReplans + 1), 3), minSafeProb, None)

        #################################### Run Simulation #############################################
        s = Run_Simulation(plan_paths, DelaysProbDictExecution, AgentLocations, GoalLocations, randGen, timestep,
                           Online_SST, inactive_agents, obstacleAgents)

        if s.runSimulation():
            print("Pass\n--------------------------------------------------------------------------")
            return (round(OfflineTime, 3), round(OnlineTime, 3), round(OfflineTime, 3) + round(OnlineTime, 3),
                    numOfReplans, s.TST, Offline_SST, round(countExpand / (numOfReplans + 1), 3), minSafeProb,
                    s.TST + SGAT)

        AgentLocations, GoalLocations = s.AgentLocations, s.remainGoals

        ########################################## Re-planning ############################################
        print("Calc New Plan")
        p, replan_time, currCountExpand = run_robust_planner_with_timeout(
            AgentLocations, GoalLocations, desired_safe_prob, DelaysProbDictExecution, mapAndDim, verifyAlpha,
            max_planning_time, obstacleAgents, Experiment, Agents_to_remove, Parameters, inactiveAgents=inactive_agents)

        countExpand += currCountExpand
        if p is None:
            print("Plan is None!\n--------------------------------------------------------------------------")
            return (round(OfflineTime, 3), None, None, numOfReplans + 1, None, Offline_SST, round(countExpand / (numOfReplans + 2), 3),
                    minSafeProb, None)

        plan_paths, _, currSafeProb, _ = p
        minSafeProb = min(minSafeProb, round(currSafeProb, 4))
        OnlineTime += replan_time
        numOfReplans += 1
        timestep = s.timestep
        Online_SST = s.TST
        SGAT += (len(GoalLocations) * replan_time)


####################################################### run Tests #################################################################################

def run_instances():
    randGen = random.Random(seedPerInstanceForDelay)
    for instance in range(instanceRange[0], instanceRange[1] + 1):
        records = []

        AgentsLocations, GoalsLocations = read_locs_from_file(instance)
        mapAndDim["nxGraph"] = GraphG(mapAndDim["Rows"], mapAndDim["Cols"], mapAndDim["Map"], AgentsLocations,
                                      GoalsLocations)

        delaysProbDictForExecution = {}
        for a in range(len(AgentsLocations)):
            delaysProbDictForExecution[a] = round(randGen.uniform(0.1, 0.95), 3)

        if experiment in ["ConsiderAgentsWithoutDelay", "ConsiderAllAgents"]:
            print(f"{experiment} - M: {mapName}, A: {len(AgentsLocations)}, G: {len(GoalsLocations)}, I: {instance}")
            result = run_Test(AgentsLocations, GoalsLocations, delaysProbDictForExecution, experiment, None, None)
            record = build_record(instance, experiment, result)
            records.append(record)

        if experiment in utilityFuncDict or experiment in ["RemoveRandomAgents", "RemoveAgentsByDelay"]:
            for agents_to_remove in range(1, 8):
                print(f"{experiment} and {agents_to_remove} agents remove,"
                      f" M: {mapName}, A: {len(AgentsLocations)}, G: {len(GoalsLocations)}, I: {instance}")

                result = run_Test(AgentsLocations, GoalsLocations, delaysProbDictForExecution, experiment,
                                  agents_to_remove, parameters)
                record = build_record(instance, f"{experiment}_{agents_to_remove}", result)
                records.append(record)

        with open(f"Output_files/Output_{configStr}.csv", mode="a", newline="", encoding="utf-8") as file:
            writerRecord = csv.writer(file)
            writerRecord.writerows(records)


def build_record(instance, experiment_name, result):
    (offlineRuntime, onlineRuntime, runtime, needReplan, sstOnline, sstOffline, countExpand, minSafeProb, sgat) = result

    return [mapName, desired_safe_prob, num_of_agents, num_of_goals, instance, experiment_name, obstacleAgents,
            offlineRuntime, onlineRuntime, runtime, sgat, needReplan, sstOnline, sstOffline, countExpand, minSafeProb]


run_instances()
