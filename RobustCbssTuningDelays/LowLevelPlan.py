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
    def __init__(self, dict_of_map_and_dim, AgentLocations, dict_cost_for_Heuristic_value):
        self.MapAndDims = dict_of_map_and_dim
        self.AgentLocations = AgentLocations
        self.dict_cost_for_Heuristic_value = dict_cost_for_Heuristic_value

    def runLowLevelPlan(self, Node, agent_that_need_update_path):
        for agent in agent_that_need_update_path:
            sequence = Node.sequence["Allocations"][agent]

            # If no allocations are present
            if len(sequence) == 1:
                Node.paths[agent]["path"] = [self.AgentLocations[agent]]
                continue

            # Decrease the previous path cost of the current agent
            Node.g -= Node.paths[agent]["cost"]

            findPath = False
            OpenList = []
            visited = {}

            S = State(self.AgentLocations[agent], sequence=[self.AgentLocations[agent]], t=0)
            heapq.heappush(OpenList, (self.calc_cost_for_Heuristic_value(S, sequence), S))

            while OpenList:
                _, S = heapq.heappop(OpenList)

                if (S.CurLocation, tuple(S.sequence), S.t) in visited:
                    continue
                visited[(S.CurLocation, tuple(S.sequence), S.t)] = True

                if len(S.sequence) == len(sequence):
                    findPath = True
                    break

                for Sl in self.GetNeighbors(S, agent, visited, Node, sequence):
                    if not visited.get((Sl.CurLocation, tuple(Sl.sequence), Sl.t), False):
                        heapq.heappush(OpenList, (self.calc_cost_for_Heuristic_value(Sl, sequence) + Sl.g, Sl))

            if not findPath:
                return False

            # Extract the path from the final goal back to the start
            Node.paths[agent] = extractPath(S)
            Node.g += S.g
        return True

    ########################################################## calc cost for Heuristic value #####################################################
    def calc_cost_for_Heuristic_value(self, S, sequence):
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

    def GetNeighbors(self, state, agent, visited, Node, sequence):
        neighbors = []
        loc = state.CurLocation
        stay = False

        # Define movement candidates for the agent
        direction_moves = (loc + 1, loc + self.MapAndDims["Cols"], loc - 1, loc - self.MapAndDims["Cols"])

        for loc_after_move in direction_moves:
            canMove = self.validateMove(loc_after_move, agent, state, Node)

            if canMove == 1:
                if loc_after_move == sequence[len(state.sequence)] and state.sequence == sequence[:len(state.sequence)]:
                    afterMoveStateSequence = state.sequence + [sequence[len(state.sequence)]]
                else:
                    afterMoveStateSequence = state.sequence[:]

                neighbors.append(State(loc_after_move, state.g + (len(sequence) - len(state.sequence)), state, afterMoveStateSequence, state.t+1))

            # Stay in the same place but increment g (cost)
            if canMove == -1 and not stay:
                stay = True
                neighbors.append(State(loc, state.g + (len(sequence) - len(state.sequence)), state, state.sequence[:], state.t+1))
                del visited[(loc, tuple(state.sequence), state.t)]

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
                return -1

        for agent1, agent2, x, t1, t2 in Node.posConstraints[agent]:
            if agent1 == agent and t1 == state.t + 1 and (
                    x != loc_after_move and x != frozenset((loc, loc_after_move))):
                return 0

            elif agent2 == agent and t2 == state.t + 1 and (
                    x != loc_after_move and x != frozenset((loc, loc_after_move))):
                return 0

        return 1
