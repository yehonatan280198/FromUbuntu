import math
from collections import defaultdict

import networkx as nx


def normalize_by_max(score):
    mx = max(score.values(), default=0.0)
    return {a: (score[a] / mx if mx > 0 else 0.0) for a in score}


class GraphG:
    def __init__(self, rows, cols, grid, agentsLoc, goalsLoc):
        self.rows = rows
        self.cols = cols
        self.grid = grid  # 0 = free, 1 = blocked
        self.agentsLoc = agentsLoc
        self.goalsLoc = goalsLoc
        self.G = self.build_graph()
        self.dictDistance = self.CalcAllDistancesFromGoals()

        self.distance = None
        self.betweenness = None
        self.connectivity = None

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
        dictDistance = defaultdict(lambda: 1000000)
        for goal in self.goalsLoc:
            distDictFromGoal = nx.single_source_shortest_path_length(self.G, goal)
            for loc, dist in distDictFromGoal.items():
                dictDistance[(loc, goal)] = dist

        return dictDistance

    def frameworkForUtility(self, parameters):
        a, b, c = parameters
        self.distance = self.compute_agents_tfidf() if a != 0 else None
        # self.betweenness = self.compute_agents_betweenness() if b != 0 else None
        self.connectivity = self.compute_agents_connectivity()

    def compute_agents_connectivity(self):
        score = {}
        G_without_Agents = self.G.copy()
        G_without_Agents.remove_nodes_from(self.agentsLoc)
        base_cc = nx.number_connected_components(G_without_Agents)

        for a in self.agentsLoc:
            H2 = G_without_Agents.copy()
            H2.add_node(a)
            for nb in self.neighbors_free(a):
                if nb in H2:
                    H2.add_edge(a, nb)
            new_cc = nx.number_connected_components(H2)
            score[a] = base_cc > new_cc

        return score

    def compute_agents_betweenness(self):
        score = {}
        betweenness = nx.betweenness_centrality(self.G, normalized=True)
        for a in self.agentsLoc:
            score[a] = betweenness.get(a)

        return normalize_by_max(score)

    def compute_idf_mean_over_min(self):
        idf = {}
        for a in self.agentsLoc:
            dists = [self.dictDistance[(a, g)] for g in self.goalsLoc]
            d_min = min(dists)
            d_mean = sum(dists) / len(dists)
            idf[a] = math.log10(d_mean / d_min)
        return idf

    def calcTotalInverseDistToGoal(self):
        totalInvDist = {}
        for g in self.goalsLoc:
            dists = [(1 / self.dictDistance[(a, g)]) for a in self.agentsLoc]
            totalInvDist[g] = sum(dists)

        return totalInvDist

    def compute_agents_tfidf(self):
        idf = self.compute_idf_mean_over_min()
        totalInvDist = self.calcTotalInverseDistToGoal()
        score = defaultdict(float)
        for a in self.agentsLoc:
            for g in self.goalsLoc:
                invDist = 1 / self.dictDistance[(a, g)]
                tf = invDist / totalInvDist[g]
                score[a] += idf[a] * tf

        return normalize_by_max(score)

    def neighbors_free(self, cell):
        neighbors = []
        r, c = divmod(cell, self.cols)

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                nidx = nr * self.cols + nc
                if self.grid[nidx] == 0:
                    neighbors.append(nidx)

        return neighbors

