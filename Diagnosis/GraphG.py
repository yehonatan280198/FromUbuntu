import math
from collections import defaultdict

import networkx as nx
import numpy as np


def normalize_by_max(score):
    mx = max(score.values(), default=0.0)
    return {a: (score[a] / mx if mx > 0 else 0.0) for a in score}


class GraphG:
    def __init__(self, mapAndDim, agentsLoc, goalsLoc):
        self.rows = mapAndDim["Rows"]
        self.cols = mapAndDim["Cols"]
        self.grid = mapAndDim["Map"]  # 0 = free, 1 = blocked
        self.freeCells = mapAndDim["FreeCells"]
        self.agentsLoc = agentsLoc
        self.goalsLoc = goalsLoc
        self.G = self.build_graph()
        self.dictDistance = defaultdict(lambda: math.inf)

    def build_graph(self) -> nx.Graph:
        G = nx.Graph()
        for idx in range(self.rows * self.cols):
            if self.grid[idx] == 0:
                G.add_node(idx)

        for idx in G.nodes:
            r, c = divmod(idx, self.cols)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    nidx = nr * self.cols + nc
                    if self.grid[nidx] == 0:
                        G.add_edge(idx, nidx)
        return G

    def CalcAllDistancesFromGoals(self):
        for goal in self.goalsLoc:
            distDictFromGoal = nx.single_source_shortest_path_length(self.G, goal)
            for loc, dist in distDictFromGoal.items():
                self.dictDistance[(loc, goal)] = dist

    def all_goals_reachable_by_at_least_one_agent(self, active_agentsLoc, inactive_agentsLoc):
        G_without_inactive_agents = self.G.copy()
        G_without_inactive_agents.remove_nodes_from(inactive_agentsLoc)
        for goal in self.goalsLoc:
            reachable = nx.node_connected_component(G_without_inactive_agents, goal)
            if not reachable.intersection(active_agentsLoc):
                return False

        return True

    def compute_distance_features(self, removed_agent_idx):

        removed_node = self.agentsLoc[removed_agent_idx]
        lengths = nx.single_source_shortest_path_length(self.G, removed_node)

        agent_distances = [lengths[node] for i, node in enumerate(self.agentsLoc) if i != removed_agent_idx and node in lengths]
        agent_distances = np.array(agent_distances)

        min_agent = float(np.min(agent_distances)) / self.freeCells
        mean_agent = float(np.mean(agent_distances)) / self.freeCells

        goal_distances = [lengths[node] for node in self.goalsLoc if node in lengths]
        goal_distances = np.array(goal_distances)

        min_goal = float(np.min(goal_distances)) / self.freeCells
        mean_goal = float(np.mean(goal_distances)) / self.freeCells

        return min_agent, mean_agent, min_goal, mean_goal

    def removed_agent_is_articulation_point(self, removed_agent_idx):
        removed_node = self.agentsLoc[removed_agent_idx]
        return int(removed_node in nx.articulation_points(self.G))

    def build_graph_after_removal(self, removed_agent_idx):
        removed_node = self.agentsLoc[removed_agent_idx]
        G2 = self.G.copy()
        if removed_node in G2:
            G2.remove_node(removed_node)
        return G2

    def removed_agent_num_components_after_removal(self, removed_agent_idx):
        G2 = self.build_graph_after_removal(removed_agent_idx)
        return nx.number_connected_components(G2)

    def removed_agent_largest_component_ratio(self, removed_agent_idx):
        G2 = self.build_graph_after_removal(removed_agent_idx)

        if G2.number_of_nodes() == 0:
            return 0.0

        largest_cc_size = max(len(cc) for cc in nx.connected_components(G2))
        return largest_cc_size / G2.number_of_nodes()

    def removed_agent_has_goal_component_without_other_agents(self, removed_agent_idx):
        G2 = self.build_graph_after_removal(removed_agent_idx)
        removed_node = self.agentsLoc[removed_agent_idx]

        other_agents = set(self.agentsLoc) - {removed_node}
        valid_goals = [g for g in self.goalsLoc if g in G2]

        for cc in nx.connected_components(G2):
            cc_set = set(cc)

            goals_in_cc = any(g in cc_set for g in valid_goals)
            agents_in_cc = any(a in cc_set for a in other_agents)

            if goals_in_cc and not agents_in_cc:
                return 1

        return 0

    def removed_agent_betweenness_centrality(self, removed_agent_idx):
        removed_node = self.agentsLoc[removed_agent_idx]
        bc = nx.betweenness_centrality(self.G, normalized=True)
        return bc.get(removed_node, 0.0)

    def compute_radius_features(self, removed_agent_idx):
        features = {}
        removed_node = self.agentsLoc[removed_agent_idx]

        radii = (2, 4)
        max_r = max(radii)
        lengths = nx.single_source_shortest_path_length(self.G, removed_node, cutoff=max_r)

        for r in radii:
            nodes_in_r = {node for node, dist in lengths.items() if dist <= r}

            num_close_agents = sum(1 for pos in self.agentsLoc if pos != removed_node and pos in nodes_in_r)

            num_close_goals = sum(1 for goal in self.goalsLoc if goal in nodes_in_r)

            local_free_space = len(nodes_in_r) - 1
            local_free_space = max(local_free_space, 0)

            features[f"Close Agents r={r}"] = num_close_agents / len(self.agentsLoc)
            features[f"Close Goals r={r}"] = num_close_goals / len(self.goalsLoc)
            features[f"Local Free Space r={r}"] = local_free_space / self.freeCells
            features[f"Crowdedness Score r={r}"] = (num_close_agents / local_free_space if local_free_space > 0 else 0.0)
            features[f"Goal Density r={r}"] = (num_close_goals / local_free_space if local_free_space > 0 else 0.0)

        return features

