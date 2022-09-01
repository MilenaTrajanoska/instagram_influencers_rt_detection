from collections import Counter

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def read_graph(
        edges_path,
        edge_weight,
        source,
        target,
        cutoff_point=0.0
):
    edges = pd.read_csv(edges_path)
    edges.dropna(axis=0, inplace=True)
    edges = edges[edges[edge_weight] > cutoff_point]
    graph = nx.from_pandas_edgelist(edges, source=source, target=target, edge_attr=edge_weight, create_using = nx.DiGraph)

    return graph, edges


def plot_graph(
        graph,
        graph_name,
        node_size=15,
        draw_labels=False,
        figure_size=(50, 50),
        alpha=0.3,
        line_style='solid',
        line_width=3,
        draw_arrows=True,
        arrow_style='-|>',
        font_color='black',
        font_size=10,
):
    plt.figure(figsize=figure_size)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=node_size)
    nx.draw_networkx_edges(graph, pos, alpha=alpha, style=line_style, arrows=draw_arrows, width=line_width, arrowstyle=arrow_style)
    if draw_labels:
        nx.draw_networkx_labels(graph, pos, font_color=font_color, font_size=font_size)
    plt.title(f'Visual representation of network: {graph_name}')
    plt.show()


def plot_degree_distribution(
        node_degrees,
        graph_name,
        figure_size=(20, 15),
        color_palette='Blues_d'
):
    plt.figure(figsize=figure_size)
    node_degrees = [d[1] for d in node_degrees]
    counts = Counter(node_degrees)
    degrees = list(counts.keys())
    values = list(counts.values())
    sns.barplot(degrees, values, palette=color_palette)
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


def get_graph_statistics_and_visualizations(
        graph,
        graph_name,
        node_size=15,
        draw_labels=False,
        figure_size=(50, 50),
        alpha=0.3,
        line_style='solid',
        line_width=3,
        draw_arrows=True,
        arrow_style='-|>',
        font_color='black',
        font_size=10,
        color_palette='Blues_d'
):
    num_nodes = graph.number_of_nodes()
    print(f'The graph has {num_nodes} nodes')
    num_edges = graph.number_of_edges()
    print(f'The graph has {num_edges} edges')
    node_degrees = graph.degree(graph.nodes())
    print(f'The graph has a maximum node degree of {max([d[1] for d in node_degrees])}')
    node_in_degrees = graph.in_degree(graph.nodes())
    print(f'The graph has a maximum in-degree of {max([d[1] for d in node_in_degrees])}')
    node_out_degrees = graph.out_degree(graph.nodes())
    print(f'The graph has a maximum out-degree of {max([d[1] for d in node_out_degrees])}')
    # num_connected_components = nx.number_connected_components(graph)
    # print(f'The number of connected components in the graph is: {num_connected_components}')
    avg_clustering_coef = nx.average_clustering(graph)
    print(f'The average clustering coefficient of the graph is: {avg_clustering_coef}')
    # diameter = get_graph_diameter(graph)
    # print(f'The graph has a diameter of: {diameter}')
    # num_nodes_in_largest_component = len(max(nx.connected_components(graph), key=len))
    # print(f'The largest connected component in the graph has {num_nodes_in_largest_component} nodes')

    plot_graph(
        graph,
        graph_name,
        node_size,
        draw_labels,
        figure_size,
        alpha,
        line_style,
        line_width,
        draw_arrows,
        arrow_style,
        font_color,
        font_size,
    )
    plot_degree_distribution(
        node_degrees,
        graph_name,
        figure_size,
        color_palette
    )
    plot_degree_distribution(
        node_in_degrees,
        graph_name + ' in-degree',
        figure_size,
        color_palette
    )
    plot_degree_distribution(
        node_out_degrees,
        graph_name + ' out-degree',
        figure_size,
        color_palette
    )
