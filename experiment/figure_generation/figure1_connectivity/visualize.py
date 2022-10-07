"""
This is third step of figure1. It generates a graph with node and edges.
"""

# pip install networkx -i https://pypi.tuna.tsinghua.edu.cn/simple
import pickle

import networkx as nx
import matplotlib.pyplot as plt


class GraphVisualization:

    def __init__(self):
        self.visual = []

    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    def visualize(self):
        plt.figure(num=1, figsize=(8, 6), dpi=500)
        graph = nx.Graph()
        graph.add_edges_from(self.visual)
        node_color_list = []
        for _ in range(len(graph.nodes)):
            node_color_list.append("#2aacd4")
        node_color_list[0] = "#fc0303"
        nx.draw_networkx(graph, with_labels=False, font_size=8,
                         pos=nx.spring_layout(graph),
                         node_size=30, node_color=node_color_list, width=0.3)
        plt.savefig("figure1_connectivity.pdf")
        plt.show()


def main():
    # input
    input_file_name = f"./connectivity_graph_0.pkl"
    with open(input_file_name, 'rb') as handle:
        edge_set = pickle.load(handle)

    graph = GraphVisualization()
    for source_node_idx in edge_set:
        target_list = edge_set[source_node_idx]
        for target_node_idx in target_list:
            if source_node_idx == target_node_idx:
                continue
            graph.addEdge(source_node_idx, target_node_idx)
    graph.visualize()


if __name__ == '__main__':
    main()
