import os
import csv
import ast
import sys
from multiprocessing import Process, Queue, Value
import gurobipy as gp
import ctypes
from kBestSequencingByService import kBestSequencingByService


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
configStr = f"{mapName}_num_of_agents_{num_of_agents}num_of_goals{num_of_goals}"

instances = 20

####################################################### Write the header of a CSV file ############################################################
if not os.path.exists("Scalability_Milp_Test_files"):
    os.makedirs("Scalability_Milp_Test_files")

columns = ["Map", "Number of agents", "Number of goals", "Instance", "Runtime", "Total variables", "Constraints"]

with open(f"Scalability_Milp_Test_files/Output_{configStr}.csv", mode="w", newline="", encoding="utf-8") as file:
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
def run_Test(queue, AgentLocations, GoalLocations, TimeToOptimize):
    reset_gurobi_model(gurobiModel)
    kBestSolver = kBestSequencingByService(AgentLocations, GoalLocations, mapAndDim, gurobiModel, TimeToOptimize)
    print([gurobiModel.NumVars, gurobiModel.NumConstrs])
    next(kBestSolver)
    queue.put([gurobiModel.NumVars, gurobiModel.NumConstrs])

####################################################### run Tests #################################################################################

def run_instances():
    for instance in range(1, instances + 1):
        AgentsLocations, GoalsLocations = read_locs_from_file(instance)
        print(f"map: {mapName}, agents: {AgentsLocations}, goals: {GoalsLocations}")

        q = Queue()
        timeToOptimize = Value(ctypes.c_double, 0.0)
        p = Process(target=run_Test, args=(q, AgentsLocations, GoalsLocations, timeToOptimize))
        p.start()
        p.join(timeout=300)

        if p.is_alive():
            p.terminate()
            p.join()
            record = [mapName, num_of_agents, num_of_goals, instance, 300, None, None]
        else:
            numVars, numConstrs = q.get()
            record = [mapName, num_of_agents, num_of_goals, instance, round(timeToOptimize.value, 3), numVars, numConstrs]

        print(f"instance {instance} is writing to CSV...\n")
        with open(f"Scalability_Milp_Test_files/Output_{configStr}.csv", mode="a", newline="", encoding="utf-8") as file:
            writerRecord = csv.writer(file)
            writerRecord.writerow(record)


run_instances()
