from dataclasses import dataclass
from ..sentence import SentenceChain
from ..elastic.elastic import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ChainCluster:
    cluster_label: int
    chains: list[SentenceChain]
    _similarity_sorted_indices: list[int]
    centroid: np.ndarray

    #---------------------------------------------------------------------------

    def __init__(self, chains: list[SentenceChain], cluster_label: int):

        self.cluster_label = cluster_label
        self.chains = chains

        chain_matrix = np.array([chain.vector for chain in chains])

        if cluster_label >= 0:
            #Calculate centroid
            self.centroid = np.average(chain_matrix, axis=0)

            #Calculate most similar chains
            sims = np.sum(cosine_similarity(chain_matrix), axis=1)
            self._similarity_sorted_indices = sorted(range(len(chains)), key=lambda i: sims[i], reverse=True)
        else:
            self.centroid = None
            self._similarity_sorted_indices = None

    #---------------------------------------------------------------------------

    def kth_most_similar_chain(self, k: int = 0) -> SentenceChain | None:
        '''
        Parameters
        ---
        k: int
            It's the k
        
        Returns
        ----
        chain: SentenceChain | None
            The k-th sentence in decreasing order of total similarity to the remaining sentences.
            Total similarity is calculated by summing the sentence's similarity scores with every other sentence
        '''
        if self._similarity_sorted_indices is not None:
            return self.chains[self._similarity_sorted_indices[k]]
        else:
            return None
        
    #---------------------------------------------------------------------------
        
    def __getitem__(self, i: int) -> SentenceChain:
        return self.chains[i]
    
    #---------------------------------------------------------------------------
        
    def __iter__(self):
        return iter(self.chains)
    
    #---------------------------------------------------------------------------

    def text(self):
        return "\n\n".join([c.text for c in self.chains])

    #---------------------------------------------------------------------------
    
    @property
    def doc(self) -> Document:
        return self.chains[0].doc