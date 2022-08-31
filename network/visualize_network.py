from network.util import read_graph, get_graph_statistics_and_visualizations

if __name__ == '__main__':
    graph, _ = read_graph('../data/healthy_food_posts/edge_data.csv', 'weight', 'user_owner', 'user_other')
    edges, nodes = graph.edges(), graph.nodes()
    get_graph_statistics_and_visualizations(graph, 'Instagram healthy food network')