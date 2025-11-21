class Run_Simulation:

    def __init__(self, plan, delaysProb, AgentLocations, GoalLocations, randGen, timestep, TST):
        self.plan = plan
        self.delaysProb = delaysProb
        self.AgentLocations = AgentLocations
        self.remainGoals = GoalLocations
        self.randGen = randGen
        self.TST = TST
        self.timestep = timestep

    def Check_Potential_Conflict(self):
        potential_locs = []

        for currAgent, path in self.plan.items():
            currLoc = path["path"][0]
            for agent, loc in potential_locs:
                if currLoc == loc and currAgent != agent:
                    return False
            potential_locs.append((currAgent, currLoc))

            if len(path["path"]) > 1:
                nextLoc = path["path"][1]
                for agent, loc in potential_locs:
                    if nextLoc == loc and currAgent != agent:
                        return False
                potential_locs.append((currAgent, nextLoc))

        return True

    def runSimulation(self):
        # Initialize the set of active agents (agents that have not finished their path)
        active_agents = {agent for agent, path in self.plan.items() if len(path["path"]) > 1}

        while active_agents:
            if not self.Check_Potential_Conflict():
                return False

            self.timestep += 1
            new_locs = []
            # Set to track agents that have completed their paths
            finish_agents = set()

            for agent, path in self.plan.items():
                # Simulate agent movement with a delay probability
                if len(path["path"]) != 1 and self.randGen.random() > self.delaysProb[agent]:
                    # Remove the first step if the agent moves
                    path["path"].pop(0)

                new_locs.append(path["path"][0])

                # If the agent has reached its destination, mark it for removal
                if len(path["path"]) == 1:
                    finish_agents.add(agent)

            active_agents -= finish_agents
            self.AgentLocations = new_locs

            remainGoalsBeforeStep = len(self.remainGoals)
            self.remainGoals = list(set(self.remainGoals) - set(new_locs))
            self.TST += self.timestep * (remainGoalsBeforeStep - len(self.remainGoals))

        return True
