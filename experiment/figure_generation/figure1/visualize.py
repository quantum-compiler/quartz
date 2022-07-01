# pip install networkx -i https://pypi.tuna.tsinghua.edu.cn/simple
import networkx as nx
import matplotlib.pyplot as plt


class GraphVisualization:

    def __init__(self):
        self.visual = []

    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    def visualize(self):
        graph = nx.Graph()
        graph.add_edges_from(self.visual)
        nx.draw_networkx(graph)
        plt.show()


def main():
    graph = GraphVisualization()
    graph.addEdge("graph", 2)
    graph.addEdge(1, 2)
    graph.addEdge(1, 3)
    graph.addEdge(5, 3)
    graph.addEdge(3, 4)
    graph.addEdge(1, 0)
    graph.visualize()


if __name__ == '__main__':
    main()
