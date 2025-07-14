from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import cached_property

from ..sentence import SentenceChain
from ..elastic import Document

#==========================================================================================================

class ChainCluster:
    '''
    A cluster of chains from a specific document
    '''
    label: int
    chains: list[SentenceChain]
    centroid: np.ndarray
    pooling_method: str
    clustering_context: ChainClustering

    VALID_METHODS = ["average", "max", "most_similar", "k_most_similar"]
    EXEMPLAR_BASED_METHODS = ["most_similar", "k_most_similar"]

    #---------------------------------------------------------------------------

    def __init__(self, chains: list[SentenceChain], cluster_label: int, pooling_method: str = "average", *, normalize: bool = True):
        '''
        A cluster of chains from a specific document
        '''
        self.label = cluster_label
        self.chains = chains
        self.clustering_context = None

        if pooling_method not in ChainCluster.VALID_METHODS:
            raise ValueError(f"Invalid pooling method {pooling_method}")
        self.pooling_method = pooling_method

        if cluster_label >= 0:
            self.centroid = ChainCluster.pooling(chains, pooling_method, normalize=normalize)
        else:
            self.centroid = None

    #---------------------------------------------------------------------------
            
    @staticmethod
    def pooling_average(chains: list[SentenceChain], *, normalize: bool = True) -> np.ndarray:
        vec = np.average(np.row_stack([c.vector for c in chains]), axis=0)
        return vec / np.linalg.norm(vec) if normalize else vec
    
    @staticmethod
    def pooling_max(chains: list[SentenceChain], *, normalize: bool = True) -> np.ndarray:
        vec = np.max(np.row_stack([c.vector for c in chains]), axis=0)
        return vec / np.linalg.norm(vec) if normalize else vec
    
    @staticmethod
    def pooling_most_similar(chains: list[SentenceChain], *, normalize: bool = True) -> np.ndarray:
        chain_matrix = [c.vector for c in chains]
        sims = np.sum(cosine_similarity(chain_matrix), axis=1)
        most_similar_index = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[0]
        vec = chains[most_similar_index]
        return vec / np.linalg.norm(vec) if normalize else vec

    @staticmethod
    def pooling(chains: list[SentenceChain], pooling_method: str, *, normalize: bool = True) -> np.ndarray:
        '''
        Arguments
        ---
        chains: list[SentenceChain]
            The chains we want to calculate the representative for
        pooling_method: str
            The method to use to generate the representative
        normalize: bool
            Normalize the representative after pooling. Defaults to ```True```

        Returns
        ---
        vec: ndarray
            The representative
        '''

        #But, there was nothing to pool
        if len(chains) == 1:
            return chains[0].vector

        match pooling_method:
            case "average": return ChainCluster.pooling_average(chains, normalize=normalize)
            case "max": return ChainCluster.pooling_max(chains, normalize=normalize)
            case "most_similar": return ChainCluster.pooling_most_similar(chains, normalize=normalize)

    #---------------------------------------------------------------------------

    @property
    def vector(self):
        '''
        Get the representative vector of this cluster
        '''
        return self.centroid

    @cached_property
    def text(self) -> str:
        return "\n\n".join([c.text for c in self.chains])
    
    @property
    def doc(self) -> Document:
        return self.chains[0].doc
    
    @property
    def id(self) -> str:
        return f"{self.doc.id:04}_{self.label:02}"
    
    #---------------------------------------------------------------------------

    def chain_matrix(self) -> np.ndarray:
        '''
        Converts the cluster into a matrix, where each row is a chain. Order is maintained
        '''
        return np.array([x.vector for x in self.chains])
    
    #---------------------------------------------------------------------------
        
    def __getitem__(self, i: int) -> SentenceChain:
        return self.chains[i]
        
    def __iter__(self):
        return iter(self.chains)

    def __len__(self) -> int:
        return len(self.chains)
    
    #---------------------------------------------------------------------------

    def data(self) -> dict:
        return {
            'id': self.doc.id,
            'label': self.label,
            'centroid': self.centroid,
            'pooling_method': self.pooling_method,
            'chains': [
                c.data() for c in self.chains
            ]
        }
    
    #---------------------------------------------------------------------------

    @classmethod
    def from_data(cls, data: dict, doc: Document) -> 'ChainCluster':
        obj = cls.__new__(cls)
        obj.chains = [SentenceChain.from_data(chain_data, doc, parent=obj) for chain_data in data['chains']]
        obj.label = data['label']
        obj.centroid = data['centroid']
        obj.pooling_method = data.get('pooling_method', "average")

        return obj
    
#====================================================================================================
    
@dataclass
class ChainClustering():
    '''
    Represents the clustering input data and results for one single document
    '''
    chains: list[SentenceChain]
    labels: list[int]
    clusters: dict[int, ChainCluster]

    def __post_init__(self):
        for cluster in self.clusters.values():
            cluster.clustering_context = self

    def data(self) -> int:
        return [c.data() for c in self.clusters.values()]
    
    def __iter__(self):
        return iter(self.clusters.values())