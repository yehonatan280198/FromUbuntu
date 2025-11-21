import math
from collections import defaultdict, deque
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class kBestSequencingByService:

    def __init__(self, AgentLocations, GoalLocations, dict_of_map_and_dim, gurobiModel):
        self.num_agents, self.num_goals = len(AgentLocations), len(GoalLocations)
        self.nodes_dict = {"All": AgentLocations + GoalLocations, "Total": self.num_agents + self.num_goals}
        self.goal_indices = list(range(self.num_agents, self.nodes_dict["Total"]))

        self.MapAndDims = dict_of_map_and_dim

        self.cost_dict = self.precompute_costs(GoalLocations)

        # Create the MILP model with a minimization objective
        self.model = gurobiModel

        # Binary variables x[i,j]: whether there is a path from node i to node j
        self.x = self.model.addVars(
            [(i, j) for i in range(self.nodes_dict["Total"]) for j in self.goal_indices if i != j],
            vtype=GRB.BINARY,
            name="x"
        )

        # Integer variables t[j]: service time at goal j
        self.t = self.model.addVars(
            self.goal_indices,
            vtype=GRB.INTEGER,
            lb=0,
            name="t"
        )

        # Objective: minimize the total service time across all goals
        self.model.setObjective(gp.quicksum(self.t[j] for j in self.goal_indices), GRB.MINIMIZE)

        # Constraints
        for j in self.goal_indices:

            # Constraint 1: each goal must have exactly one incoming edge (visited once)
            self.model.addConstr(gp.quicksum(self.x[i, j] for i in range(self.nodes_dict["Total"]) if i != j) == 1)

            # Constraint 2: each goal can lead to at most one other goal
            self.model.addConstr(gp.quicksum(self.x[j, k] for k in self.goal_indices if k != j) <= 1)

        # Constraint 3: each agent starts at most one path
        for a in range(self.num_agents):
            self.model.addConstr(gp.quicksum(self.x[a, j] for j in self.goal_indices) <= 1)

        # Constraint 4: timing constraints based on transitions
        M = 1000000
        for i in range(self.nodes_dict["Total"]):
            for j in self.goal_indices:
                if i != j:
                    cost = self.cost_dict.get((self.nodes_dict["All"][i], self.nodes_dict["All"][j]))
                    if i < self.num_agents:
                        # If coming directly from an agent to a goal
                        self.model.addConstr(self.t[j] >= cost - (1 - self.x[i, j]) * M)
                        self.model.addConstr(self.t[j] <= cost + (1 - self.x[i, j]) * M)
                    else:
                        # If coming from a previous goal to the current goal
                        self.model.addConstr(self.t[j] >= self.t[i] + cost - (1 - self.x[i, j]) * M)
                        self.model.addConstr(self.t[j] <= self.t[i] + cost + (1 - self.x[i, j]) * M)

    def __iter__(self):
        return self

    def __next__(self):
        self.model.optimize()

        if self.model.status == GRB.INFEASIBLE or (self.model.status == GRB.TIME_LIMIT and self.model.SolCount == 0):
            return {"Allocations": {}, "Cost": math.inf}

        current_edges = {(i, j) for (i, j) in self.x if self.x[i, j].X > 0.5}
        service_times = {j: round(self.t[j].X) for j in self.goal_indices}

        paths = {}
        for a in range(self.num_agents):
            curr_path = [self.nodes_dict["All"][a]]
            current = a
            while True:
                next_node = next((j for (i, j) in current_edges if i == current), None)
                if next_node is None:
                    break
                curr_path.append(self.nodes_dict["All"][next_node])
                current = next_node
            paths[a] = curr_path

        # Add exclusion constraint to prevent repeating this edge set
        self.model.addConstr(gp.quicksum(self.x[i, j] for (i, j) in current_edges) <= len(current_edges) - 1)
        return {"Allocations": paths, "Cost": sum(service_times.values())}

    def precompute_costs(self, GoalLocations):
        precomputed_cost = defaultdict(lambda: 1000000)
        for goal in GoalLocations:
            self.BFS(goal, precomputed_cost)
        return precomputed_cost

    def BFS(self, goal, precomputed_cost):
        visited = np.zeros(self.MapAndDims["Cols"] * self.MapAndDims["Rows"], dtype=bool)
        queue = deque([(goal, 0)])

        while queue:
            current_loc, cost = queue.popleft()

            if visited[current_loc]:
                continue
            visited[current_loc] = True

            precomputed_cost[(current_loc, goal)] = cost

            for neighbor_loc, new_cost in self.get_neighbors(current_loc, cost):
                if not visited[neighbor_loc]:
                    queue.append((neighbor_loc, new_cost))

    def get_neighbors(self, current_loc, cost):
        neighbors = []
        for neighborLoc in [current_loc + 1, current_loc + self.MapAndDims["Cols"],
                            current_loc - 1, current_loc - self.MapAndDims["Cols"]]:
            if self.validate_move(neighborLoc, current_loc):
                neighbors.append((neighborLoc, cost + 1))
        return neighbors

    def validate_move(self, loc_after_move, loc):
        # Extract the agent's location and direction before taking the next step

        # If the agent is at the top or bottom boundary, it cannot move up or down
        if not (0 <= loc_after_move < self.MapAndDims["Cols"] * self.MapAndDims["Rows"]):
            return False

        # If the agent is at the right boundary, it cannot move right
        if loc % self.MapAndDims["Cols"] == self.MapAndDims["Cols"] - 1 and loc_after_move % \
                self.MapAndDims["Cols"] == 0:
            return False

        # If the agent is at the left boundary, it cannot move left
        if loc % self.MapAndDims["Cols"] == 0 and loc_after_move % self.MapAndDims["Cols"] == \
                self.MapAndDims["Cols"] - 1:
            return False

        if self.MapAndDims["Map"][loc_after_move] != 0:
            return False

        return True
