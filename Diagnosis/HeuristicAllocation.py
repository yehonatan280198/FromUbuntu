import math


class HeuristicAllocation:
    def __init__(self, AgentLocations, GoalLocations, dict_of_map_and_dim, delayProbDict, obsAgents, inactiveAgents, dictDistance):
        self.AgentLocations = AgentLocations
        self.GoalLocations = GoalLocations
        self.MapAndDims = dict_of_map_and_dim
        self.inactiveAgents = inactiveAgents
        self.obsAgentLocs = [] if obsAgents == False else [l for a, l in enumerate(AgentLocations) if a in inactiveAgents]
        self.cost_dict = dictDistance
        self.delayProbDict = delayProbDict

        self.forbidden_pair_alloc = {i + 1: set() for i in range(len(self.GoalLocations))}
        self.currIndex = len(self.GoalLocations) + 1
        self.allALloc = False
        self.alloc = set()

    def __iter__(self):
        return self

    def __next__(self):
        if self.allALloc:
            return {"Allocations": {}, "Cost": math.inf}

        while True:
            remaining_goals = set(self.GoalLocations)

            agents_dict = {agent: {"alloc": [init_loc], "sst": 0, "dist": 0} for agent, init_loc
                           in enumerate(self.AgentLocations)}

            curr_alloc = {}

            for index in range(1, len(self.GoalLocations) + 1):
                best_agent, best_goal, best_sst = None, None, float("inf")

                for agent, agent_data in agents_dict.items():
                    if agent in self.inactiveAgents:
                        continue
                    cur_loc = agent_data["alloc"][-1]
                    agent_dist = agent_data["dist"]

                    for goal in remaining_goals:
                        if (agent, goal) in self.forbidden_pair_alloc[index]:
                            continue

                        sst = agent_dist + self.cost_dict[(cur_loc, goal)]
                        if sst < best_sst:
                            best_sst, best_agent, best_goal = sst, agent, goal

                if best_goal is None:
                    self.allALloc = True
                    return {"Allocations": {}, "Cost": math.inf}

                remaining_goals.remove(best_goal)
                agents_dict[best_agent]["alloc"].append(best_goal)
                agents_dict[best_agent]["sst"] += best_sst
                agents_dict[best_agent]["dist"] = best_sst
                curr_alloc[index] = (best_agent, best_goal)

            self.makeConstraint(curr_alloc)

            key = frozenset((agent, tuple(data["alloc"])) for agent, data in agents_dict.items())

            if key in self.alloc:
                continue

            self.alloc.add(key)

            alloc = {agent: data["alloc"] for agent, data in agents_dict.items()}
            cost = sum(data["sst"] for data in agents_dict.values())

            return {"Allocations": alloc, "Cost": cost}

    def makeConstraint(self, currAlloc):
        for key, value in reversed(list(self.forbidden_pair_alloc.items())):
            if len(value) == len(self.AgentLocations) * (len(self.GoalLocations) - key + 1) - 1:
                self.forbidden_pair_alloc[key] = set()
            else:
                self.forbidden_pair_alloc[key].add(currAlloc[key])
                return
        self.allALloc = True

# mat = {"Rows": 10, "Cols": 10, "Map": [0 for _ in range(10 * 10)]}
# agents = [4, 94]
# goals = [24, 14, 34]
#
# h = HeuristicAllocation(agents, goals, mat)
# for i in range(1, 30):
#     print(f"{i}: {next(h)}")
