import math
import time
from collections import defaultdict
from queue import PriorityQueue
from multiprocessing import Process, Queue, Value
import ctypes


from FindConflict import FindConflict
from LowLevelPlan import LowLevelPlan
from NodeStateClasses import Node
from kBestSequencingByMakespan import kBestSequencingByMakespan
from Verify import Verify
from kBestSequencingByService import kBestSequencingByService
from kBestSequencingBySoc import kBestSequencingBySoc


class RobustPlanner:
    def __init__(self, AgentLocations, GoalLocations, desired_safe_prob, delaysProb, MapAndDims, verifyAlpha,
                 gurobiModel, process_queue, typeOfVerify, countExpand, optimize):
        self.AgentLocations = AgentLocations
        self.desired_safe_prob = desired_safe_prob
        self.OPEN = PriorityQueue()
        self.Num_roots_generated = 0
        self.K_optimal_sequences = {}
        self.final_sol = None
        self.process_queue = process_queue
        self.delaysProb = delaysProb
        self.countExpand = countExpand
        self.optimize = optimize

        if self.optimize == "SST":
            self.K_Best_Seq_Solver = kBestSequencingByService(self.AgentLocations, GoalLocations, MapAndDims, gurobiModel)
        if self.optimize == "SOC":
            self.K_Best_Seq_Solver = kBestSequencingBySoc(self.AgentLocations, GoalLocations, MapAndDims, gurobiModel)
        if self.optimize == "MAKESPAN":
            self.K_Best_Seq_Solver = kBestSequencingByMakespan(self.AgentLocations, GoalLocations, MapAndDims, gurobiModel)

        self.LowLevelPlanner = LowLevelPlan(MapAndDims, self.AgentLocations, self.K_Best_Seq_Solver.cost_dict, optimize)
        self.findConflict_algorithm = FindConflict(delaysProb)
        self.verify_algorithm = Verify(delaysProb, desired_safe_prob, verifyAlpha, self.process_queue, self.findConflict_algorithm, typeOfVerify)

    ####################################################### run ############################################################

    def run(self):
        print("New Plan", flush=True)
        # Calculate the best sequence of task allocations (k=1)
        self.K_optimal_sequences[1] = next(self.K_Best_Seq_Solver)
        if self.K_optimal_sequences[1]['Cost'] == math.inf:
            print(f"Allocation not found", flush=True)
            time.sleep(60)

        # Increment root node counter
        self.Num_roots_generated += 1

        # Create the root node
        Root = Node()
        # Assign the best sequence of task allocations for all agents to the root node
        Root.sequence = self.K_optimal_sequences[1]
        # Generate paths and calculate the cost for the root node
        self.LowLevelPlanner.runLowLevelPlan(Root, list(range(len(self.AgentLocations))))

        # Add the root node to the open list
        self.OPEN.put((Root.g, Root))

        # Continue processing nodes in the open list until it is empty
        while not self.OPEN.empty():

            # Get the node with the lowest cost
            _, N = self.OPEN.get()
            # Check if a new root needs to be generated
            N = self.CheckNewRoot(N)
            if N is None:
                continue

            self.countExpand.value += 1

            # If the paths in the current node are verified as valid, avoiding collisions with probability P, return them as the solution
            if not N.isPositiveNode and self.verify_algorithm.verify(N):
                if self.desired_safe_prob == "NotAvailable":
                    self.process_queue.put([dict(N.paths), N.g, self.desired_safe_prob])
                return

            # Identify the first conflict in the paths
            conflict = self.findConflict_algorithm.findConflict(N)

            if conflict is None:
                continue
            else:
                _, _, _, x, agent1AndTime, agent2AndTime = conflict

            # Generate child nodes with constraints to resolve the conflict and add child nodes to the open list
            if agent1AndTime[1] != 0:
                A1 = self.GenChild(N, (agent1AndTime[0], x, agent1AndTime[1]))
                if A1 is not None:
                    self.OPEN.put((A1.g, A1))

            if agent2AndTime[1] != 0:
                A2 = self.GenChild(N, (agent2AndTime[0], x, agent2AndTime[1]))
                if A2 is not None:
                    self.OPEN.put((A2.g, A2))

            if self.delaysProb[0] != 0 and max(agent1AndTime[1], agent2AndTime[1]) != 1:
                A3 = self.GenChild(N, (agent1AndTime[0], agent2AndTime[0], x, agent1AndTime[1], agent2AndTime[1]))
                self.OPEN.put((A3.g, A3))

        return None
    ####################################################### Check new root ############################################################

    def CheckNewRoot(self, N):
        # If the current node cost is within the threshold of the current optimal sequence
        if N.g <= self.K_optimal_sequences[self.Num_roots_generated]["Cost"]:
            return N

        # Generate a new root with an updated sequence
        self.Num_roots_generated += 1
        self.K_optimal_sequences[self.Num_roots_generated] = next(self.K_Best_Seq_Solver)

        if self.K_optimal_sequences[self.Num_roots_generated]["Cost"] == math.inf:
            return N

        # Create a new root node
        newRoot = Node()
        newRoot.sequence = self.K_optimal_sequences[self.Num_roots_generated]
        # Calculate paths and cost for the new root
        self.LowLevelPlanner.runLowLevelPlan(newRoot, list(range(len(self.AgentLocations))))

        self.OPEN.put((newRoot.g, newRoot))
        self.OPEN.put((N.g, N))
        return None

    ####################################################### Get conflict ############################################################

    def GenChild(self, N, NewCons):
        A = Node()
        A.negConstraints = defaultdict(set,
                                       {agent: constraints.copy() for agent, constraints in N.negConstraints.items()})
        A.posConstraints = defaultdict(set,
                                       {agent: constraints.copy() for agent, constraints in N.posConstraints.items()})
        A.paths = {
            agent: {"path": list(info["path"]), "cost": info["cost"]}
            for agent, info in N.paths.items()
        }
        A.sequence = N.sequence
        A.g = N.g

        if len(NewCons) == 3:
            agent, _, _ = NewCons
            A.negConstraints[agent].add(NewCons)
            if not self.LowLevelPlanner.runLowLevelPlan(A, [agent]):
                return None

        else:
            A.isPositiveNode = True
            agent1, agent2, _, _, _ = NewCons
            A.posConstraints[agent1].add(NewCons)
            A.posConstraints[agent2].add(NewCons)

        return A

def planner_process(AgentLocations, GoalLocations, safe_prob, DelaysProbDict, mapAndDim, verifyAlpha, gurobiModel, queue, typeOfVerify, countExpand, optimize):
    cbss = RobustPlanner(AgentLocations, GoalLocations, safe_prob, DelaysProbDict, mapAndDim, verifyAlpha, gurobiModel, queue, typeOfVerify, countExpand, optimize)
    cbss.run()

def run_robust_planner_with_timeout(AgentLocations, GoalLocations, safe_prob, DelaysProbDict, mapAndDim,
                                    verifyAlpha, gurobiModel, max_planning_time, typeOfVerify, optimize):
    queue = Queue()
    countExpand = Value(ctypes.c_long, 0)
    process = Process(
        target=planner_process,
        args=(AgentLocations, GoalLocations, safe_prob, DelaysProbDict, mapAndDim, verifyAlpha, gurobiModel, queue, typeOfVerify, countExpand, optimize)
    )
    start_time = time.time()
    process.start()
    process.join(timeout=max_planning_time)
    plan_time = time.time() - start_time

    if process.is_alive():
        print("Planning Timeout reached.")
        process.terminate()
        process.join()

    last_result = None
    while not queue.empty():
        last_result = queue.get_nowait()
    expansions = countExpand.value
    return last_result, min(60, plan_time), expansions
