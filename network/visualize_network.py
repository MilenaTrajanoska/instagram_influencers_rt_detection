from network.util import read_graph, get_graph_statistics_and_visualizations
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    graph, _ = read_graph('../data/healthy_food_posts/edge_data.csv', 'weight', 'user_owner', 'user_other')
    edges, nodes = graph.edges(), graph.nodes()
    get_graph_statistics_and_visualizations(
        graph,
        'Instagram healthy food network',
        node_size=10,
        figure_size=(150, 150),
        alpha=0.3,
        line_width=0.2,
    )