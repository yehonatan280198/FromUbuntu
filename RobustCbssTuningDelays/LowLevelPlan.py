import heapq
from NodeStateClasses import State


########################################################## Extract path #####################################################3
def extractPath(state):
    agent_path_and_cost = {"path": [], "cost": state.g}
    while state is not None:
        agent_path_and_cost["path"].insert(0, state.CurLocation)
        state = state.parent
    return agent_path_and_cost


########################################################## LowLevelPlan Class #####################################################3

class LowLevelPlan:
    def __init__(self, dict_of_map_and_dim, AgentLocations, dict_cost_for_Heuristic_value, optimize):
        self.MapAndDims = dict_of_map_and_dim
        self.AgentLocations = AgentLocations
        self.dict_cost_for_Heuristic_value = dict_cost_for_Heuristic_value
        self.optimize = optimize

    def runLowLevelPlan(self, Node, agent_that_need_update_path):
        for agent in agent_that_need_update_path:
            sequence = Node.sequence["Allocations"][agent]

            # If no allocations are present
            if len(sequence) == 1:
                Node.paths[agent]["path"] = [self.AgentLocations[agent]]
                continue

            # Decrease the previous path cost of the current agent
            if self.optimize != "MAKESPAN":
                Node.g -= Node.paths[agent]["cost"]
            else:
                Node.g = max((data["cost"] for agent_id, data in Node.paths.items() if agent_id != agent), default=0)

            findPath = False
            OpenList = []
            visited = {}

            S = State(self.AgentLocations[agent], sequence=[self.AgentLocations[agent]], t=0)
            if self.optimize == "SST":
                heapq.heappush(OpenList, (self.calc_sst_for_Heuristic_value(S, sequence), S))
            else:
                heapq.heappush(OpenList, (self.calc_soc_or_makespan_for_Heuristic_value(S, sequence), S))

            while OpenList:
                _, S = heapq.heappop(OpenList)

                if (S.CurLocation, tuple(S.sequence), S.t) in visited:
                    continue
                visited[(S.CurLocation, tuple(S.sequence), S.t)] = True

                if len(S.sequence) == len(sequence):
                    findPath = True
                    break

                for Sl in self.GetNeighbors(S, agent, Node, sequence):
                    if not visited.get((Sl.CurLocation, tuple(Sl.sequence), Sl.t), False):
                        if self.optimize == "SST":
                            heapq.heappush(OpenList, (self.calc_sst_for_Heuristic_value(Sl, sequence) + Sl.g, Sl))
                        else:
                            heapq.heappush(OpenList, (self.calc_soc_or_makespan_for_Heuristic_value(Sl, sequence) + Sl.g, Sl))

            if not findPath:
                return False

            # Extract the path from the final goal back to the start
            Node.paths[agent] = extractPath(S)
            if self.optimize != "MAKESPAN":
                Node.g += S.g
            else:
                Node.g = max(S.g, Node.g)
        return True

    ########################################################## calc cost for Heuristic value #####################################################
    def calc_sst_for_Heuristic_value(self, S, sequence):
        if len(S.sequence) == len(sequence):
            return 0

        steps = 0
        current_loc = S.CurLocation
        total_service_time = 0

        for i in range(len(S.sequence), len(sequence)):
            steps += self.dict_cost_for_Heuristic_value[(current_loc, sequence[i])]
            total_service_time += steps
            current_loc = sequence[i]

        return total_service_time

    def calc_soc_or_makespan_for_Heuristic_value(self, S, sequence):
        if len(S.sequence) == len(sequence):
            return 0

        current_loc = S.CurLocation
        soc_remaining = 0

        for i in range(len(S.sequence), len(sequence)):
            soc_remaining += self.dict_cost_for_Heuristic_value[(current_loc, sequence[i])]
            current_loc = sequence[i]

        return soc_remaining

    def GetNeighbors(self, state, agent, Node, sequence):
        neighbors = []
        loc = state.CurLocation

        # Define movement candidates for the agent
        direction_moves = (loc + 1, loc + self.MapAndDims["Cols"], loc - 1, loc - self.MapAndDims["Cols"])

        for loc_after_move in direction_moves:
            canMove = self.validateMove(loc_after_move, agent, state, Node)

            if canMove == 1:
                if loc_after_move == sequence[len(state.sequence)] and state.sequence == sequence[:len(state.sequence)]:
                    afterMoveStateSequence = state.sequence + [sequence[len(state.sequence)]]
                else:
                    afterMoveStateSequence = state.sequence[:]

                step_cost = (len(sequence) - len(state.sequence)) if self.optimize == "SST" else 1
                neighbors.append(State(loc_after_move, state.g + step_cost, state, afterMoveStateSequence, state.t + 1))

        if self.validateMove(loc, agent, state, Node) == 1:
            step_cost = (len(sequence) - len(state.sequence)) if self.optimize == "SST" else 1
            neighbors.append(State(loc, state.g + step_cost, state, state.sequence[:], state.t + 1))

        return neighbors

    ########################################################## validate Move #####################################################
    def validateMove(self, loc_after_move, agent, state, Node):
        # Extract the agent's location and direction before taking the next step
        loc = state.CurLocation
        cols, rows = self.MapAndDims["Cols"], self.MapAndDims["Rows"]
        max_cells = cols * rows

        # If the agent is at the top or bottom boundary, it cannot move up or down
        if not (0 <= loc_after_move < max_cells):
            return 0

        col_loc = loc % cols
        col_after = loc_after_move % cols

        if (col_loc == 0 and col_after == cols - 1) or (col_loc == cols - 1 and col_after == 0):
            return 0

        if self.MapAndDims["Map"][loc_after_move] != 0:
            return 0

        # Check if the move violates any negative constraints
        for z, x, t in Node.negConstraints[agent]:
            if t == state.t + 1 and (x == loc_after_move or x == frozenset((loc, loc_after_move))):
                return 0

        for agent1, agent2, x, t1, t2 in Node.posConstraints[agent]:
            if agent1 == agent and t1 == state.t + 1 and (
                    x != loc_after_move and x != frozenset((loc, loc_after_move))):
                return 0

            elif agent2 == agent and t2 == state.t + 1 and (
                    x != loc_after_move and x != frozenset((loc, loc_after_move))):
                return 0

        return 1
