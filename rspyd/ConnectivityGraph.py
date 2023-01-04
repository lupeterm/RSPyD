from typing import List


class ConnectivityGraph:
    def __init__(self, num_nodes) -> None:
        self.graph_indices = []
        for _ in range(num_nodes):
            self.graph_indices.append([0,0])
        self.group_indices = [0]*num_nodes
        self.graph: List[int] = []

    def add_node(self, home, neighbors):
        self.graph_indices[home][0] = len(self.graph)
        self.graph_indices[home][1] = len(neighbors)
        self.graph += neighbors

    def get_num_neighbors(self, node):
        return self.graph_indices[node][1] - self.graph_indices[node][0]

    def get_neighbors(self, node: int) -> List[int]:
        n = self.graph_indices[node]
        return self.graph[n[0]: n[0] + n[1]]
