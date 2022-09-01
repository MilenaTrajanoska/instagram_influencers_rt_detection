from difussion.threshold import ThresholdModel
from network.util import read_graph

if __name__ == '__main__':
    graph, _ = read_graph('../data/healthy_food_posts/edge_data.csv', 'weight', 'user_owner', 'user_other')
    model = ThresholdModel(graph, 0.9)
    influencers, influence_dict = model.calculate_optimal_influencer_set_greedy(5)
    print(influencers)
    print(influence_dict)