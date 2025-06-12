from ..elastic import Document
from ..clustering import ChainCluster, ChainClustering
from ..sentence import Sentence, SentenceChain

from dataclasses import dataclass, field

#==========================================================================================================

@dataclass
class ProcessedDocument():
    '''
    The preprocessing result (embeddings, chains and clusters) for a specific document.
    This is loaded from a database after we first retrieve the relevant docs from Elasticsearch
    '''
    doc: Document
    clustering: ChainClustering
    sentences: list[Sentence]
    params: dict = field(default=None)

    @property
    def chains(self) -> list[SentenceChain]:
        return self.clustering.chains
    
    @property
    def labels(self) -> list[int]:
        return self.clustering.labels
    
    @property
    def clusters(self) -> dict[int, ChainCluster]:
        return self.clustering.clusters

#==========================================================================================================