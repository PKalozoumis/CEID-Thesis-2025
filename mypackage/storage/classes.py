from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..elastic import Document
    from ..clustering import ChainCluster, ChainClustering
    from ..sentence import Sentence, SentenceChain

#==========================================================================================================

class ProcessedDocument():
    '''
    The preprocessing result (embeddings, chains and clusters) for a specific document.
    This is loaded from a database after we first retrieve the relevant docs from Elasticsearch
    '''
    doc: Document
    clustering: ChainClustering
    sentences: list[Sentence]
    params: dict

    def __init__(self, doc: Document, clustering: ChainClustering, sentences: list[Sentence], params: dict = None):
        self.doc = doc
        self.clustering = clustering
        self.sentences = sentences
        self.params = params

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