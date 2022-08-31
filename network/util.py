from collections import Counter

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def read_graph(edges_path, edge_weight, source, target, cutoff_point=0.0):
    edges = pd.read_csv(edges_path)
    edges.dropna(axis=0, inplace=True)
    edges = edges[edges[edge_weight] != 0]
    if cutoff_point > 0:
        edges = edges[edges[edge_weight] >= edges[edge_weight].quantile(q=cutoff_point)]
    graph = nx.from_pandas_edgelist(edges, source=source, target=target, edge_attr=edge_weight)

    return graph, edges


def plot_graph(graph, graph_name, node_size=15, draw_labels=False):
    plt.figure(figsize=(50, 50))
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=node_size)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, style='solid', arrows=True, width=3, arrowstyle='-|>')
    if draw_labels:
        nx.draw_networkx_labels(graph, pos, font_color='red', font_size=8)
    plt.title(f'Visual representation of network: {graph_name}')
    plt.show()


def plot_degree_distribution(node_degrees, graph_name):
    plt.figure(figsize=(20, 15))
    node_degrees = [d[1] for d in node_degrees]
    counts = Counter(node_degrees)
    degrees = list(counts.keys())
    values = list(counts.values())
    sns.barplot(degrees, values, palette="Blues_d")
    plt.title(f'Visual representation of node degrees for the network: {graph_name}')
    plt.show()


def get_graph_diameter(graph):
    """

      Returns nx.diameter(graph) if the graph is connected
      else returns the maximum of the shortest paths between each pair of nodes

      graph: networkx.Graph

      Returns: float - diameter

    """
    if nx.is_connected(graph):
        return nx.diameter(graph)
    return max([max(j.values()) for (i, j) in nx.shortest_path_length(graph)])


def get_graph_statistics_and_visualizations(graph, graph_name):
    num_nodes = graph.number_of_nodes()
    print(f'The graph has {num_nodes} nodes')
    num_edges = graph.number_of_edges()
    print(f'The graph has {num_edges} edges')
    node_degrees = graph.degree(graph.nodes())
    print(f'The graph has a maximum node degree of {max(node_degrees)}')
    num_connected_components = nx.number_connected_components(graph)
    print(f'The number of connected components in the graph is: {num_connected_components}')
    avg_clustering_coef = nx.average_clustering(graph)
    print(f'The average clustering coefficient of the graph is: {avg_clustering_coef}')
    diameter = get_graph_diameter(graph)
    print(f'The graph has a diameter of: {diameter}')
    num_nodes_in_largest_component = len(max(nx.connected_components(graph), key=len))
    print(f'The largest connected component in the graph has {num_nodes_in_largest_component} nodes')

    plot_graph(graph, graph_name=graph_name)
    plot_degree_distribution(node_degrees, graph_name=graph_name)