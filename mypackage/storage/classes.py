from ..elastic import Document
from ..clustering import ChainCluster, ChainClustering
from ..sentence import Sentence, SentenceChain

from dataclasses import dataclass, field

#==========================================================================================================

@dataclass
class ProcessedDocument():
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
    

#Things relating to the document itself will be retrieved either from elasticsearch or from the cache
#   Document has already been retrieved. Also, we do not care about its type. Could be Document or ElasticDocument
#We should not store text, filter_path/text_path/etc or the sentences themselves
#We do store the id, so that we can point back to the document
#The sentences will be broken down upon retrieval
#...but we do need to store their offsets and embeddings
#...as well as the chains that have formed (with their representative, and all other information)
#Same thing with the chain clusters: we need to store everything

#==========================================================================================================