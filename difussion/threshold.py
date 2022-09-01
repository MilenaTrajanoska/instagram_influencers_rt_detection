import numpy as np


class ThresholdModel:
    def __init__(self, graph, cutoff_point):
        self.graph = graph
        self.cutoff_point = cutoff_point
        self.nodes = graph.nodes()
        self.influencers = self._get_influencers()
        self.sig_a_init = 0

    def _get_influencers(self):
        out_degrees = self.graph.out_degree(self.nodes)
        degrees_only = [d[1] for d in out_degrees]
        min_degree = np.quantile(degrees_only, self.cutoff_point)
        return [d[0] for d in out_degrees if d[1] >= min_degree]

    def _propagate_influence(self, nodes, influence_dict):
        if not len(nodes):
            return

        influenced_nodes = []
        for node in nodes:
            for neighbor in self.graph.neighbors(node):
                if self.graph.in_degree(neighbor, weight='weight') > 0:
                    influence_dict[neighbor] = 1
                    influenced_nodes.append(neighbor)

        self._propagate_influence(influenced_nodes, influence_dict)

    def _begin_influence(self, influencers):
        influence_dict = {k: 0 if k not in influencers else 1 for k in list(self.nodes)}

        self._propagate_influence(influencers, influence_dict)
        return influence_dict

    def calculate_optimal_influencer_set_greedy(self, k):
        if k <= 0:
            raise ValueError('The number of influencers to select should be greater than 0')

        if k == len(self.influencers):
            return self._begin_influence(self.influencers)

        influence_dict = self._begin_influence(self.influencers)
        self.sig_a_init = len([v for v in influence_dict.values() if v == 1])

        influencer_to_remove = None
        min_influence = len(self.nodes)
        max_influence = -1
        max_influence_dict = influence_dict.copy()

        while len(self.influencers) > k:
            for node in self.influencers:
                sub_influencers = [n for n in self.influencers if n != node]
                influence_dict = self._begin_influence(sub_influencers)
                sig_a_prim = len([v for v in influence_dict.values() if v == 1])
                if sig_a_prim < min_influence:
                    min_influence = sig_a_prim
                    influencer_to_remove = node
                if sig_a_prim > max_influence:
                    max_influence = sig_a_prim
                    max_influence_dict = influence_dict.copy()
            if max_influence == 0 or not influencer_to_remove:
                return self.influencers, self._begin_influence(self.influencers)

            else:
                self.influencers.remove(influencer_to_remove)

        return self.influencers, max_influence_dict
