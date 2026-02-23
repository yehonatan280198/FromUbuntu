import math
import random
from itertools import combinations
from scipy.stats import norm


class Verify:

    def __init__(self, delaysProb, safe_prob, verifyAlpha, findConflictALg, obstacles_agents, inactiveAgents, process_queue):
        self.obstacles_agents = obstacles_agents
        self.delaysProb = delaysProb
        self.desired_safe_prob = safe_prob
        self.verifyAlpha = verifyAlpha
        self.curr_sol = [None, None, -math.inf]
        self.process_queue = process_queue
        self.randGen = random.Random(47)
        self.findConflictALg = findConflictALg
        self.inactiveAgents = inactiveAgents

    ############################################### Verify ####################################################
    def verify(self, N):
        # Check if 1-Robust
        if not self.findConflictALg.Check_Potential_Conflict_in_first_step(N):
            return False

        return self.anytime_verify(N)

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
        count_success = self.run_s_simulations(s0, N)

        while True:
            P0 = count_success / s0
            p_c1, p_c2 = self.compute_safe_prob_bounds(P0, s0)

            if p_c1 > self.curr_sol[2]:
                self.curr_sol = [dict(N.paths), N.g, p_c1, self.inactiveAgents]
                self.process_queue.put(self.curr_sol)

                if p_c1 >= self.desired_safe_prob:
                    return True

            if p_c2 < self.desired_safe_prob:
                return False

            count_success += self.run_s_simulations(1, N)
            s0 += 1

    def compute_confidence_bounds(self, curr_safe_prob, s0):
        margin = (norm.ppf(1 - self.verifyAlpha)) * math.sqrt((curr_safe_prob * (1 - curr_safe_prob)) / s0)
        return curr_safe_prob + margin, curr_safe_prob - margin

    def required_simulations(self, curr_safe_prob):
        return math.ceil(((norm.ppf(1 - self.verifyAlpha)) ** 2) * (curr_safe_prob / (1 - curr_safe_prob)))

    ############################################### Run Simulation ####################################################
    def run_s_simulations(self, s0, N):
        count_success = 0

        # Run s0 simulations
        for sim in range(s0):
            # Create a copy of the paths for independent simulation
            paths_copy = {
                agent: {"path": list(info_path["path"]), "cost": info_path["cost"]}
                for agent, info_path in N.paths.items()
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
                    if agent in self.inactiveAgents and not self.obstacles_agents:
                        continue
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
