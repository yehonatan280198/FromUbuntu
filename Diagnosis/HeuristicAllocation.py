from collections import defaultdict, deque
import numpy as np
import math


class HeuristicAllocation:
    def __init__(self, AgentLocations, GoalLocations, dict_of_map_and_dim):
        self.AgentLocations = AgentLocations
        self.GoalLocations = GoalLocations
        self.MapAndDims = dict_of_map_and_dim
        self.cost_dict = self.precompute_costs(GoalLocations)

        self.forbidden_first_pairs = set()

    def __iter__(self):
        return self

    def __next__(self):
        remaining_goals = set(self.GoalLocations)
        agents_dict = {agent: {"alloc": [initLoc], "sst": 0, "dist": 0} for agent, initLoc in
                       enumerate(self.AgentLocations)}

        step_index = 0
        first_pair = None

        while remaining_goals:
            bestAgent, bestGoal, bestDist, bestSst = None, None, None, float("inf")

            for agent, agent_data in agents_dict.items():
                curLoc = agent_data["alloc"][-1]
                agentDist = agent_data["dist"]

                for goal in remaining_goals:
                    if step_index == 0 and (agent, goal) in self.forbidden_first_pairs:
                        continue

                    goalServiceTime = agentDist + self.cost_dict[(curLoc, goal)]

                    if goalServiceTime < bestSst:
                        bestSst = goalServiceTime
                        bestAgent = agent
                        bestGoal = goal
                        bestDist = goalServiceTime

            if bestAgent is None or bestGoal is None:
                return {"Allocations": {}, "Cost": math.inf, "InactiveAgents": []}

            remaining_goals.remove(bestGoal)
            agents_dict[bestAgent]["alloc"].append(bestGoal)
            agents_dict[bestAgent]["sst"] += bestSst
            agents_dict[bestAgent]["dist"] = bestDist

            if step_index == 0:
                first_pair = (bestAgent, bestGoal)

            step_index += 1

        alloc, cost, inactiveAgents = {}, 0, set()
        for agent, data in agents_dict.items():
            alloc[agent] = data["alloc"]
            cost += data["sst"]
            if len(alloc[agent]) == 1:
                inactiveAgents.add(agent)

        self.forbidden_first_pairs.add(first_pair)

        return {"Allocations": alloc, "Cost": cost, "InactiveAgents": inactiveAgents}

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


mat = {"Rows": 10, "Cols": 10, "Map": [0 for _ in range(10 * 10)]}
agents = [4, 94]
goals = [24, 34, 14]

h = HeuristicAllocation(agents, goals, mat)
