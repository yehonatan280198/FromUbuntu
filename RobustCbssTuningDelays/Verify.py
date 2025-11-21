import math
import random
from itertools import combinations
from scipy.stats import norm


def verifyWithoutDelay(paths):
    for agent1, agent2 in combinations(paths.keys(), 2):
        path1 = paths[agent1]
        path2 = paths[agent2]

        # Create loc-time dictionaries
        locTimes1 = set()
        for i, loc in enumerate(path1["path"]):
            locTimes1.add((i, loc))

        locTimes2 = set()
        for i, loc in enumerate(path2["path"]):
            locTimes2.add((i, loc))

        # Detect location conflicts
        common_locs = locTimes1 & locTimes2
        if len(common_locs) != 0:
            return False

        # Create edge-time dictionaries
        edgeTimes1 = set()
        for i in range(len(path1["path"]) - 1):
            if path1["path"][i] != path1["path"][i + 1]:
                edge = (path1["path"][i], path1["path"][i + 1])
                edgeTimes1.add((i + 1, edge))

        edgeTimes2 = set()
        for i in range(len(path2["path"]) - 1):
            if path2["path"][i] != path2["path"][i + 1]:
                edge = (path2["path"][i], path2["path"][i + 1])
                edgeTimes2.add((i + 1, edge))

        # Detect edge conflicts, including reversed edges
        for time1, edge1 in edgeTimes1:
            reversed_edge1 = (edge1[1], edge1[0])
            if (time1, reversed_edge1) in edgeTimes2:
                return False

    return True


class Verify:

    def __init__(self, delaysProb, safe_prob, verifyAlpha, process_queue, findConflictALg, typeOfVerify):
        self.delaysProb = delaysProb
        self.desired_safe_prob = safe_prob
        self.verifyAlpha = verifyAlpha
        self.randGen = random.Random(47)
        self.findConflictALg = findConflictALg
        self.curr_sol = [None, None, -math.inf]
        self.process_queue = process_queue
        self.typeOfVerify = typeOfVerify

    ############################################### Verify ####################################################
    def verify(self, N):
        # Check if 1-Robust
        if not self.findConflictALg.Check_Potential_Conflict_in_first_step(N):
            return False

        if self.delaysProb[0] == 0:
            return verifyWithoutDelay(N.paths)

        if self.typeOfVerify == "Strict":
            return self.strict_verify(N)
        else:
            return self.anytime_verify(N)

    ############################################### Strict Verify ####################################################
    def strict_verify(self, N):
        s0 = max(30, self.required_simulations(self.desired_safe_prob))
        count_success = self.run_s_simulations(s0, N.paths)

        while True:
            P0 = count_success / s0
            c1, c2 = self.compute_confidence_bounds(self.desired_safe_prob, s0)

            if P0 >= c1:
                self.process_queue.put([dict(N.paths), N.g, self.desired_safe_prob])
                return True

            if P0 < c2:
                return False

            s0 += 1
            count_success += self.run_s_simulations(1, N.paths)

    def compute_confidence_bounds(self, curr_safe_prob, s0):
        margin = (norm.ppf(1 - self.verifyAlpha)) * math.sqrt((curr_safe_prob * (1 - curr_safe_prob)) / s0)
        return curr_safe_prob + margin, curr_safe_prob - margin

    def required_simulations(self, curr_safe_prob):
        return math.ceil(((norm.ppf(1 - self.verifyAlpha)) ** 2) * (curr_safe_prob / (1 - curr_safe_prob)))

    ############################################### Anytime Verify ####################################################
    def compute_safe_prob_bounds(self, P0, s0):
        z = norm.ppf(1 - self.verifyAlpha)
        A = s0 + z ** 2
        B = -(2 * s0 * P0 + z ** 2)
        C = s0 * P0 ** 2

        discriminant = B ** 2 - 4 * A * C
        sqrt_discriminant = math.sqrt(discriminant)

        p1 = (-B + sqrt_discriminant) / (2 * A)
        p2 = (-B - sqrt_discriminant) / (2 * A)

        return [min(max(p, 0), 1) for p in sorted([p1, p2])]

    def anytime_verify(self, N):
        s0 = max(30, self.required_simulations(self.desired_safe_prob))
        count_success = self.run_s_simulations(s0, N.paths)

        while True:
            P0 = count_success / s0
            p_c1, p_c2 = self.compute_safe_prob_bounds(P0, s0)

            if p_c1 > self.curr_sol[2]:
                self.curr_sol = [dict(N.paths), N.g, p_c1]
                self.process_queue.put(self.curr_sol)

                if p_c1 >= self.desired_safe_prob:
                    return True

            if p_c2 < self.desired_safe_prob:
                return False

            count_success += self.run_s_simulations(1, N.paths)
            s0 += 1

    ############################################### Run Simulation ####################################################
    def run_s_simulations(self, s0, paths):
        count_success = 0

        # Run s0 simulations
        for sim in range(s0):
            # Create a copy of the paths for independent simulation
            paths_copy = {
                agent: {"path": list(info_path["path"]), "cost": info_path["cost"]}
                for agent, info_path in paths.items()
            }

            # Initialize the set of active agents (agents that have not finished their path)
            active_agents = {agent for agent, info_path in paths_copy.items() if len(info_path["path"]) > 1}
            # Flag to indicate if a collision occurs
            collision = False

            while active_agents:
                # Set to track current agent locations
                locsAndEdge = set()
                # Set to track agents that have completed their paths
                finish_agents = set()

                for agent, info_path in paths_copy.items():
                    # Current path of the agent
                    lastLoc = info_path["path"][0]

                    # Simulate agent movement with a delay probability
                    if len(info_path["path"]) != 1 and self.randGen.random() > self.delaysProb[agent]:
                        # Remove the first step if the agent moves
                        info_path["path"].pop(0)

                    # Current location of the agent
                    loc = info_path["path"][0]
                    # Check for collision
                    if loc in locsAndEdge or (loc, lastLoc) in locsAndEdge:
                        collision = True
                        break
                    locsAndEdge.add(loc)
                    locsAndEdge.add((lastLoc, loc))

                    # If the agent has reached its destination, mark it for removal
                    if len(info_path["path"]) == 1:
                        finish_agents.add(agent)

                # Stop the simulation if a collision occurs
                if collision:
                    break

                # Remove agents that have completed their paths
                active_agents -= finish_agents

            # Increment success count if no collision occurred
            if not collision:
                count_success += 1

        # Return the number of successful simulations
        return count_success
