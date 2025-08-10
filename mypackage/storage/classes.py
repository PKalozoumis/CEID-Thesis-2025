from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, overload
from more_itertools import always_iterable
import os
import pickle
import shutil

from ..clustering import ChainCluster, ChainClustering
from ..elastic import Document, Session, ElasticDocument
from ..sentence import Sentence, SentenceChain
from ..helper import DEVICE_EXCEPTION

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

class DatabaseSession(ABC):

    @overload
    def load(sess: Session, docs: int|ElasticDocument) -> ProcessedDocument: ...

    @overload
    def load(sess: Session, docs: list[int]|list[ElasticDocument]) -> list[ProcessedDocument]: ...

    @abstractmethod
    def load(self, sess: Session, docs: int|list[int]|ElasticDocument|list[ElasticDocument]):
        pass

    @abstractmethod
    def store(self, clustering: ChainClustering, *, params: dict = None):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def delete(self):
        pass

    @abstractmethod
    def available_experiments(self, requested_experiments: str) -> list[str]:
        pass

    @abstractmethod
    def list_experiments():
        pass

    @abstractmethod
    def check_temp():
        pass

    def _restore_clusters(self, doc: Document, data: dict) -> ProcessedDocument:
        '''
        Recreates all cluster objects for a specific document using the retrieved data
        '''

        params = data['params']
        data = data['data']

        #Recreate the cluster dictionary, by mapping each label to its ChainCluster,
        #the same way the clusters are returned from chain_clustering
        clusters = {}

        #From our cluster objects, we want to get back the labels for all the chains
        offset_and_label = []
        offset_and_label: list[tuple[SentenceChain, int]]

        for cluster_data in data:
            #Recreate this cluster from its data
            cluster = ChainCluster.from_data(cluster_data, doc)
            clusters[cluster.label] = cluster
            offset_and_label.extend((chain, cluster.label) for chain in cluster)
        
        offset_and_label.sort(key=lambda tup: tup[0].first_index)
        chains = list([tup[0] for tup in offset_and_label])
        labels = list([tup[1] for tup in offset_and_label])

        #Assign index to each chain
        #(temporary, because currently the index is not stored in the pickles)
        #(maybe not so temporary after all...)
        for i, chain in enumerate(chains):
            chain.chain_index = i

        sentences = [sentence for chain in chains for sentence in chain]
        doc.sentences = sentences

        return ProcessedDocument(doc, ChainClustering(chains, labels, clusters), sentences, params)

#==========================================================================================================

class PickleSession(DatabaseSession):

    base_path: str
    sub_path: str

    def __init__(self, base_path: str | None = None, sub_path: str | None = None):
        self.base_path = base_path
        self.sub_path = sub_path

    #----------------------------------------------------

    @property
    def full_path(self) -> str:
        return os.path.join(self.base_path, self.sub_path)

    #----------------------------------------------------

    def load(self, sess: Session, docs: int|list[int]|ElasticDocument|list[ElasticDocument]):
        out = []

        for doc in always_iterable(docs):
            
            #We only passed a document id, so we have to retrieve it first
            if type(doc) is int:
                doc_obj = ElasticDocument(sess, doc, text_path="article")
            #We passed an existing document, so we use it
            else:
                doc_obj = doc

            #Retrieve pickle
            with open(os.path.join(self.full_path, f"{doc_obj.id}.pkl"), "rb") as f:
                data = pickle.load(f)

            out.append(self._restore_clusters(doc_obj, data))

        if isinstance(docs, list):
            return out
        else:
            return out[0]
        
    #----------------------------------------------------

    def store(self, clustering: ChainClustering, *, params: dict = None):
        '''
        Saves clusters of one specific document to a pickle file

        Arguments
        ---
        clusters: dict
            The clusters returned by chain_clustering
        path: str
            Path to store the pickle files in
        params: dict, optional
            The parameters used for all operations (e.g. chaining threshold, pooling methods, UMAP parameters, etc)
        '''
        out = clustering.data()

        with open(os.path.join(self.full_path, f"{out[0]['id']}.pkl"), "wb") as f:
            pickle.dump({'params': params, 'data': out}, f)

    #----------------------------------------------------

    def close(self):
        pass

    #----------------------------------------------------

    def delete(self):
        shutil.rmtree(self.full_path)

    #----------------------------------------------------

    def available_experiments(self, requested_experiments: str) -> list[str]:
        existing_names = os.listdir(self.base_path)

        if requested_experiments == "all":
            requested_experiments = ",".join(existing_names)
        else:
            for name in requested_experiments.split(','):
                if name not in existing_names:
                    raise DEVICE_EXCEPTION(f"YOU CALLED FOR '{name}', BUT NOBODY CAME")
                
        return [name for name in requested_experiments.split(",") if name in existing_names]
    
    #----------------------------------------------------
    
    def list_experiments(self) -> list[str]:
        return os.listdir(self.base_path)
    
    #----------------------------------------------------
    
    def check_temp(self) -> bool:
        return os.path.exists(os.path.join(self.full_path, ".temp"))

#==========================================================================================================

class MongoSession(DatabaseSession):

    host: str
    name: str
    collection: str

    def __init__(self, host: str = "localhost:27017", name: str | None = None, collection: str | None = None):
        self.host = host
        self.name = name
        self.collection = collection

    #----------------------------------------------------

    def load(self, sess: Session, docs: int|list[int]|ElasticDocument|list[ElasticDocument]):
        pass
        
    #----------------------------------------------------

    def store(self, clustering: ChainClustering, *, params: dict = None):
        pass

#==========================================================================================================