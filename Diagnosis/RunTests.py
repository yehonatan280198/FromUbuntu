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


def seed_from_range(key: str, base: int = 100):
    start, end = map(int, key.split("_"))
    bucket = (start - 1) // 10
    return base + bucket + 1


####################################################### Create Map #################################################################################
def create_map(map_name):
    file_path = f"OurResearch.domain/{map_name}.map"
    with open(file_path, "r") as file:
        lines = file.readlines()
    map_start_index = lines.index("map\n") + 1
    map_lines = lines[map_start_index:]

    currMap = []
    rows, cols, numOfFreeCells, numOfObs = 0, 0, 0, 0
    for line in map_lines:
        cols = len(line.strip())
        rows += 1
        for char in line.strip():
            if char == ".":
                currMap += [0]
                numOfFreeCells += 1
            else:
                currMap += [1]
                numOfObs += 1

    return {"Rows": rows, "Cols": cols, "Map": currMap, "ObsRatio": numOfObs / (rows * cols),
            "FreeCells": numOfFreeCells}


####################################################### Global Variables ######################################################################
mapName = sys.argv[1]
mapAndDim = create_map(mapName)
num_of_agents, num_of_goals = int(sys.argv[2]), int(sys.argv[3])
instanceRange = sys.argv[4]
seedPerInstanceForDelay = seed_from_range(instanceRange)
whoOfAgentsToRemove = sys.argv[5]  # 0 = Nothing, 1-25 = Agents
configStr = f"M={mapName}--A={num_of_agents}--G={num_of_goals}--I={instanceRange}--R={whoOfAgentsToRemove}"

verifyAlpha = 0.05
max_planning_time = 60
desired_safe_prob = 0.7
max_instance_time = 300

####################################################### Write the header of a CSV file ############################################################
if not os.path.exists("Output_files"):
    os.makedirs("Output_files")

columns = ["Map", "Map Width", "Map Height", "Map Obstacle Ratio", "Desired Safe prob", "Number of agents",
           "Number of goals", "Agent Density", "Instance", "Offline Runtime", "Online Runtime",
           "Runtime", "SGAT", "Num Of Replan", "Online SST", "Offline SST", "Number of Expands",
           "Min Safe Prob", "Removed Agent Delay", "Removed Agent Delay Zscore", "Removed Agent Delay Relative",
           "Removed Agent X", "Removed Agent Y", "Removed Agent Min Distance To Agents", "Removed Agent Mean Distance To Agents",
           "Removed Agent Min Distance To Goals", "Removed Agent Mean Distance To Goals", "Removed Agent Is Articulation Point",
           "Removed Agent Num Components After Removal", "Removed Agent Largest Component Ratio",
           "Removed Agent Has Goal Component Without Other Agents", "Removed Agent Betweenness Centrality",
           "Removed Agent Num Close Agents r=2", "Removed Agent Num Close Goals r=2", "Removed Agent Local Free Space r=2",
           "Removed Agent Crowdedness Score r=2", "Removed Agent Goal Density r=2", "Removed Agent Num Close Agents r=4",
           "Removed Agent Num Close Goals r=4", "Removed Agent Local Free Space r=4", "Removed Agent Crowdedness Score r=4",
           "Removed Agent Goal Density r=4"]

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


def run_Test(AgentLocations, GoalLocations, DelaysProbDictExecution, InactiveAgents, GraphObj):
    randGen = random.Random(44)
    start_time = time.time()
    minSafeProb = math.inf

    ####################################### Offline Planning #############################################
    print("Calc Plan")
    p, OfflineTime, countExpand = run_robust_planner_with_timeout(
        AgentLocations, GoalLocations, desired_safe_prob, DelaysProbDictExecution, mapAndDim, verifyAlpha,
        max_planning_time, InactiveAgents, GraphObj)

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
                           Online_SST, inactive_agents)

        if s.runSimulation():
            print("Pass\n--------------------------------------------------------------------------")
            return (round(OfflineTime, 3), round(OnlineTime, 3), round(OfflineTime, 3) + round(OnlineTime, 3),
                    numOfReplans, s.TST, Offline_SST, round(countExpand / (numOfReplans + 1), 3), minSafeProb,
                    s.TST + SGAT)

        AgentLocations, GoalLocations = s.AgentLocations, s.remainGoals
        GraphObj = GraphG(mapAndDim, AgentLocations, GoalLocations)
        GraphObj.CalcAllDistancesFromGoals()

        ########################################## Re-planning ############################################
        print("Calc New Plan")
        p, replan_time, currCountExpand = run_robust_planner_with_timeout(
            AgentLocations, GoalLocations, desired_safe_prob, DelaysProbDictExecution, mapAndDim, verifyAlpha,
            max_planning_time, InactiveAgents, GraphObj)

        countExpand += currCountExpand
        if p is None:
            print("Plan is None!\n--------------------------------------------------------------------------")
            return (round(OfflineTime, 3), None, None, numOfReplans + 1, None, Offline_SST,
                    round(countExpand / (numOfReplans + 2), 3),
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

    startInstance, endInstance = map(int, instanceRange.split("_"))
    for instance in range(startInstance, endInstance + 1):
        records = []

        AgentsLocations, GoalsLocations = read_locs_from_file(instance)

        delaysProbDictForExecution = {}
        for a in range(len(AgentsLocations)):
            delaysProbDictForExecution[a] = round(randGen.uniform(0.5, 0.95), 3)

        startRemove, endRemove = map(int, whoOfAgentsToRemove.split("_"))
        for whoRemove in range(startRemove, endRemove + 1):
            print(f"M: {mapName}, A: {len(AgentsLocations)}, G: {len(GoalsLocations)}, I: {instance}, W: {whoRemove}")
            if whoRemove == 0:
                graphObj = GraphG(mapAndDim, AgentsLocations, GoalsLocations)
                graphObj.CalcAllDistancesFromGoals()
                result = run_Test(AgentsLocations, GoalsLocations, delaysProbDictForExecution, set(), graphObj)
                record = build_record(instance, result, None)
                records.append(record)
            else:
                graphObj = GraphG(mapAndDim, AgentsLocations, GoalsLocations)
                loc = AgentsLocations[whoRemove - 1]

                # Features
                (min_agent, mean_agent, min_goal, mean_goal) = graphObj.compute_distance_features(whoRemove - 1)
                features = {
                    "Removed Agent Delay": delaysProbDictForExecution[whoRemove - 1],
                    "Removed Agent Delay Zscore": compute_removed_agent_delay_zscore(delaysProbDictForExecution, whoRemove - 1),
                    "Removed Agent Delay Relative": compute_removed_agent_delay_relative(delaysProbDictForExecution,whoRemove - 1),
                    "Removed Agent X": (loc % mapAndDim["Cols"]) / (mapAndDim["Cols"] - 1),
                    "Removed Agent Y": (loc // mapAndDim["Cols"]) / (mapAndDim["Rows"] - 1),
                    "Removed Agent Min Distance To Agents": min_agent,
                    "Removed Agent Mean Distance To Agents": mean_agent,
                    "Removed Agent Min Distance To Goals": min_goal,
                    "Removed Agent Mean Distance To Goals": mean_goal,
                    "Removed Agent Is Articulation Point": graphObj.removed_agent_is_articulation_point(whoRemove - 1),
                    "Removed Agent Num Components After Removal": graphObj.removed_agent_num_components_after_removal(whoRemove - 1),
                    "Removed Agent Largest Component Ratio": graphObj.removed_agent_largest_component_ratio(whoRemove - 1),
                    "Removed Agent Has Goal Component Without Other Agents": graphObj.removed_agent_has_goal_component_without_other_agents(whoRemove - 1),
                    "Removed Agent Betweenness Centrality": graphObj.removed_agent_betweenness_centrality(whoRemove - 1)
                } | graphObj.compute_radius_features(whoRemove - 1)

                mapAndDim["Map"][loc] = 1
                graphObj.G.remove_node(loc)
                graphObj.CalcAllDistancesFromGoals()
                result = run_Test(AgentsLocations, GoalsLocations, delaysProbDictForExecution, {whoRemove - 1},
                                  graphObj)
                record = build_record(instance, result, features)
                records.append(record)
                mapAndDim["Map"][loc] = 0

        with open(f"Output_files/Output_{configStr}.csv", mode="a", newline="", encoding="utf-8") as file:
            writerRecord = csv.writer(file)
            writerRecord.writerows(records)


def build_record(instance, result, features):
    (offlineRuntime, onlineRuntime, runtime, needReplan, sstOnline, sstOffline, countExpand, minSafeProb, sgat) = result

    if features is None:
        feature_values = [""] * 24
    else:
        feature_values = [features[name] for name in features]

    row = [mapName, mapAndDim["Cols"], mapAndDim["Rows"], mapAndDim["ObsRatio"], desired_safe_prob, num_of_agents,
           num_of_goals, num_of_agents / mapAndDim["FreeCells"], instance, offlineRuntime, onlineRuntime, runtime,
           sgat, needReplan, sstOnline, sstOffline, countExpand, minSafeProb] + feature_values

    return row


def compute_removed_agent_delay_zscore(delays_dict, removed_agent_idx):
    delays = list(delays_dict.values())
    n = len(delays)

    mean_delay = sum(delays) / n
    variance = sum((d - mean_delay) ** 2 for d in delays) / n
    std_delay = math.sqrt(variance)

    if std_delay == 0:
        return 0.0

    removed_delay = delays_dict[removed_agent_idx]
    return float((removed_delay - mean_delay) / std_delay)


def compute_removed_agent_delay_relative(delays_dict, removed_agent_idx):
    delays = list(delays_dict.values())
    mean_delay = sum(delays) / len(delays)

    if mean_delay == 0:
        return 0.0

    removed_delay = delays_dict[removed_agent_idx]
    return float(removed_delay / mean_delay)


run_instances()
