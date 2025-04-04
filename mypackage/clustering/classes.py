from dataclasses import dataclass
from ..sentence import SentenceChain
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ChainCluster:

    def __init__(self, chains: list[SentenceChain], cluster_label: int):

        self.cluster_label = cluster_label
        self.chains = chains

        chain_matrix = np.array([chain.vector for chain in chains])

        if cluster_label >= 0:
            #Calculate centroid
            self.centroid = np.average(chain_matrix, axis=0)

            #Calculate most similar chains
            sims = np.sum(cosine_similarity(chain_matrix), axis=1)
            self.similarity_sorted_indices_ = sorted(range(len(chains)), key=lambda i: sims[i], reverse=True)
        else:
            self.centroid = None
            self.similarity_sorted_indices_ = None

    def kth_most_similar_chain(self, i: int = 0):
        if self.similarity_sorted_indices_ is not None:
            return self.chains[self.similarity_sorted_indices_[i]]
        else:
            return None
        
    def __iter__(self):
        return iter(self.chains)