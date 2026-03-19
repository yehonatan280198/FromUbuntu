import math

class HeuristicAllocation:
    def __init__(self, AgentLocations, GoalLocations, inactiveAgents, graphObj):
        self.AgentLocations = AgentLocations
        self.GoalLocations = GoalLocations
        self.graphObj = graphObj
        self.cost_dict = graphObj.dictDistance

        self.active_agents = [
            agent for agent in range(len(self.AgentLocations))
            if agent not in inactiveAgents]

        self.finished = False
        self.first_built = False
        self.current_path = None
        self.seen_allocations = set()
        self.feasible = self._all_goals_reachable_from_some_active_agent()

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished or not self.feasible:
            return {"Allocations": {}, "Cost": math.inf}

        while True:
            if not self.first_built:
                path = self._build_greedy_from_prefix([])
                self.first_built = True

                if path is None:
                    self.finished = True
                    return {"Allocations": {}, "Cost": math.inf}

                self.current_path = path

            else:
                next_path = self._advance_path(self.current_path)
                if next_path is None:
                    self.finished = True
                    return {"Allocations": {}, "Cost": math.inf}
                self.current_path = next_path

            result = self._path_to_result(self.current_path)
            if result is None:
                continue

            key = self._allocation_key(result["Allocations"])
            if key in self.seen_allocations:
                continue

            self.seen_allocations.add(key)
            return result

    def _init_agents_dict(self):
        return {
            agent: {
                "alloc": [init_loc],
                "sst": 0,
                "dist": 0
            }
            for agent, init_loc in enumerate(self.AgentLocations)
        }

    def _get_sorted_choices(self, agents_dict, remaining_goals):
        choices = []

        for agent in self.active_agents:
            agent_data = agents_dict[agent]
            cur_loc = agent_data["alloc"][-1]
            agent_dist = agent_data["dist"]

            for goal in remaining_goals:
                travel = self.cost_dict[(cur_loc, goal)]
                if not math.isfinite(travel):
                    continue
                new_dist = agent_dist + travel
                choices.append((agent, goal, new_dist))

        choices.sort(key=lambda x: (x[2], x[0], x[1]))
        return choices

    def _apply_choice(self, agents_dict, remaining_goals, choice):
        agent, goal, new_dist = choice

        agents_dict[agent]["alloc"].append(goal)
        agents_dict[agent]["sst"] += new_dist
        agents_dict[agent]["dist"] = new_dist
        remaining_goals.remove(goal)

    def _build_greedy_from_prefix(self, prefix_spec):
        agents_dict = self._init_agents_dict()
        remaining_goals = set(self.GoalLocations)
        path = []

        num_steps = len(self.GoalLocations)

        for step in range(num_steps):
            choices = self._get_sorted_choices(agents_dict, remaining_goals)
            if not choices:
                return None

            if step < len(prefix_spec):
                chosen_idx = prefix_spec[step]
                if chosen_idx >= len(choices):
                    return None
            else:
                chosen_idx = 0

            choice = choices[chosen_idx]
            self._apply_choice(agents_dict, remaining_goals, choice)

            path.append({
                "choice": (choice[0], choice[1]),
                "choices": choices,
                "chosen_idx": chosen_idx
            })

        return path

    def _advance_path(self, path):
        for i in range(len(path) - 1, -1, -1):
            current_idx = path[i]["chosen_idx"]
            num_choices = len(path[i]["choices"])

            if current_idx + 1 < num_choices:
                new_prefix = [path[j]["chosen_idx"] for j in range(i)]
                new_prefix.append(current_idx + 1)

                rebuilt = self._build_greedy_from_prefix(new_prefix)
                if rebuilt is not None:
                    return rebuilt

        return None

    def _path_to_result(self, path):
        agents_dict = self._init_agents_dict()
        remaining_goals = set(self.GoalLocations)

        for node in path:
            agent, goal = node["choice"]

            agent_data = agents_dict[agent]
            cur_loc = agent_data["alloc"][-1]
            travel = self.cost_dict[(cur_loc, goal)]
            new_dist = agent_data["dist"] + travel

            if goal not in remaining_goals:
                return None

            agents_dict[agent]["alloc"].append(goal)
            agents_dict[agent]["sst"] += new_dist
            agents_dict[agent]["dist"] = new_dist
            remaining_goals.remove(goal)

        alloc = {
            agent: data["alloc"][:]
            for agent, data in agents_dict.items()
        }
        cost = sum(data["sst"] for data in agents_dict.values())

        active_agent_starts = {
            path[0] for _, path in alloc.items() if len(path) > 1
        }
        inactive_agent_starts = {
            path[0] for _, path in alloc.items() if len(path) == 1
        }

        if not self.graphObj.all_goals_reachable_by_at_least_one_agent(
                active_agent_starts,
                inactive_agent_starts
        ):
            return None

        return {
            "Allocations": alloc,
            "Cost": cost
        }

    def _allocation_key(self, alloc):
        return frozenset(
            (agent, tuple(path))
            for agent, path in alloc.items()
        )

    def _all_goals_reachable_from_some_active_agent(self):
        for goal in self.GoalLocations:
            if not any(
                    math.isfinite(self.cost_dict[(self.AgentLocations[a], goal)])
                    for a in self.active_agents
            ):
                return False
        return True


# a = [10, 50, 70]
# g = [30, 60, 90]
# m = {"Rows": 12, "Cols": 12, "Map": [0] * 12 * 12, "FreeCells": 12*12}
# graphObj = GraphG(m, a, g)
# graphObj.CalcAllDistancesFromGoals()
# h = HeuristicAllocation(a, g, {}, graphObj)
# count = 1
# while True:
#     res = h.__next__()
#     if res["Cost"] == math.inf:
#         break
#     print(f"{count}. {res}")
#     count += 1

