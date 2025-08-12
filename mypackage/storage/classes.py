from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, overload
from more_itertools import always_iterable
import os
import pickle
import shutil
from pymongo import MongoClient, collection, database
from collections import defaultdict

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

    _base_path: str
    _sub_path: str

    @property
    @abstractmethod
    def db_type(self) -> str:
        pass

    @overload
    def load(self, sess: Session, docs: int|ElasticDocument) -> ProcessedDocument: ...

    @overload
    def load(self, sess: Session, docs: list[int]|list[ElasticDocument]) -> list[ProcessedDocument]: ...

    @abstractmethod
    def load(self, sess: Session, docs: int|list[int]|ElasticDocument|list[ElasticDocument]) -> ProcessedDocument|list[ProcessedDocument]:
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
    def list_experiments(self) -> list[str]:
        pass

    @abstractmethod
    def is_temp(self) -> bool:
        pass

    @abstractmethod
    def set_temp(self):
        pass

    def _restore_clusters(self, doc: Document, data: dict) -> ProcessedDocument:
        '''
        Recreates all cluster objects for a specific document using the retrieved data
        '''
        
        if 'params' in data:
            params = data['params']
            data = data['data']
        else:
            params = None

        #Recreate the cluster dictionary, by mapping each label to its ChainCluster,
        #the same way the clusters are returned from chain_clustering
        clusters = {}

        #From our cluster objects, we want to get back the labels for all the chains
        #The chains appear in the same order as in the document
        offset_and_label = []
        offset_and_label: list[tuple[SentenceChain, int]]

        for cluster_data in data:
            #Recreate this cluster from its data
            cluster = ChainCluster.from_data(cluster_data, doc)
            clusters[cluster.label] = cluster
            offset_and_label.extend((chain, cluster.label) for chain in cluster)
        
        #Sort chains by first index to ensure they appear in the same order as in the document
        offset_and_label.sort(key=lambda tup: tup[0].first_index)

        chains = list([tup[0] for tup in offset_and_label])
        labels = list([tup[1] for tup in offset_and_label])

        '''
        #Assign index to each chain
        for i, chain in enumerate(chains):
            chain.chain_index = i
        '''
            
        sentences = [sentence for chain in chains for sentence in chain]
        doc.sentences = sentences

        return ProcessedDocument(doc, ChainClustering(chains, labels, clusters), sentences, params)

#==========================================================================================================

class PickleSession(DatabaseSession):
    _base_path: str
    _sub_path: str

    def __init__(self, base_path: str | None = None, sub_path: str | None = None):
        self._base_path = base_path
        self._sub_path = sub_path

    #----------------------------------------------------

    @property
    def db_type(self) -> str:
        return "pickle"

    #----------------------------------------------------

    @property
    def full_path(self) -> str:
        return os.path.join(self._base_path, self._sub_path)

    #----------------------------------------------------

    @property
    def base_path(self) -> str:
        return self.base_path
    
    @property
    def sub_path(self) -> str:
        return self.sub_path

    #----------------------------------------------------

    @base_path.setter
    def base_path(self, value: str):
        self._base_path = value
    
    @sub_path.setter
    def sub_path(self, value: str):
        self._sub_path = value

    #----------------------------------------------------

    @overload
    def load(self, sess: Session, docs: int|ElasticDocument) -> ProcessedDocument: ...

    @overload
    def load(self, sess: Session, docs: list[int]|list[ElasticDocument]) -> list[ProcessedDocument]: ...

    def load(self, sess: Session, docs: int|list[int]|ElasticDocument|list[ElasticDocument]) -> ProcessedDocument|list[ProcessedDocument]:
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

        os.makedirs(self.full_path, exist_ok=True)

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
        existing_names = os.listdir(self._base_path)

        if requested_experiments == "all":
            requested_experiments = ",".join(existing_names)
        else:
            #Check if the names you requested exist
            for name in requested_experiments.split(','):
                if name not in existing_names:
                    raise DEVICE_EXCEPTION(f"YOU CALLED FOR '{name}', BUT NOBODY CAME")
        
        #Return the intersection of requested names and existing names
        return [name for name in requested_experiments.split(",") if name in existing_names]
    
    #----------------------------------------------------
    
    def list_experiments(self) -> list[str]:
        return os.listdir(self._base_path)
    
    #----------------------------------------------------
    
    def is_temp(self) -> bool:
        return os.path.exists(os.path.join(self.full_path, ".temp"))
    
    #----------------------------------------------------
    
    def set_temp(self):
        os.makedirs(self.full_path, exist_ok=True)
        open(os.path.join(self.full_path, ".temp"), "w").close()

#==========================================================================================================

class MongoSession(DatabaseSession):

    host: str
    _base_path: str
    _sub_path: str
    client: MongoClient

    #----------------------------------------------------

    @property
    def db_type(self) -> str:
        return "mongo"
    
    #----------------------------------------------------

    def __init__(self, *, host: str = "localhost:27017", db_name: str | None = None, collection: str | None = None):
        self.host = host
        self._base_path = db_name
        self._sub_path = collection
        self.client = MongoClient(f"mongodb://{host}/")

    #----------------------------------------------------

    @property
    def collection(self) -> collection.Collection:
        return self.client[self.base_path][self.sub_path]
    
    @property
    def database(self) -> database.Database:
        return self.client[self.base_path]
    
    #----------------------------------------------------

    @property
    def base_path(self) -> str:
        return self._base_path
    
    @property
    def sub_path(self) -> str:
        return self._sub_path

    #----------------------------------------------------

    @base_path.setter
    def base_path(self, value: str):
        self._base_path = value
    
    @sub_path.setter
    def sub_path(self, value: str):
        self._sub_path = value

    #----------------------------------------------------

    @overload
    def load(self, sess: Session, docs: int|ElasticDocument) -> ProcessedDocument: ...

    @overload
    def load(self, sess: Session, docs: list[int]|list[ElasticDocument]) -> list[ProcessedDocument]: ...

    def load(self, sess: Session, docs: int|list[int]|ElasticDocument|list[ElasticDocument]) -> ProcessedDocument|list[ProcessedDocument]:
        out = []

        doc_objects = []
        doc_ids = []

        #Get document objects from input
        for doc in always_iterable(docs):
            #We only passed a document id, so we have to retrieve it first
            if type(doc) is int:
                doc_obj = ElasticDocument(sess, doc, text_path="article")
            #We passed an existing document, so we use it
            else:
                doc_obj = doc

            doc_objects.append(doc_obj)
            doc_ids.append(int(doc_obj.id))

        doc_to_clusters = defaultdict(list)

        #Retrieve from Mongo with only one query (?)
        cursor = self.collection.find({'id': {'$in': doc_ids}})
        for data in cursor:
            doc_to_clusters[data['id']].append(data)
    
        #Create clusters
        for doc_obj in doc_objects:
           out.append(self._restore_clusters(doc_obj, doc_to_clusters[doc_obj.id]))

        #Return results
        if isinstance(docs, list):
            return out
        else:
            return out[0]
        
    #----------------------------------------------------

    def store(self, clustering: ChainClustering, *, params: dict = None):
        '''
        Saves clusters of one specific document to a collection

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
        self.collection.insert_many(out)

    #----------------------------------------------------

    def close(self):
        self.client.close()

    #----------------------------------------------------

    def delete(self):
        self.collection.drop()
        #Delete the respective entry in the temp collection, if exists
        self.database['temp'].delete_many({self.sub_path: 1})

    #----------------------------------------------------

    def available_experiments(self, requested_experiments: str) -> list[str]:
        '''
        Returns the names of the available experiments
        '''
        existing_names = self.database.list_collection_names()

        if requested_experiments == "all":
            requested_experiments = ",".join(existing_names)
        else:
            #Check if the names you requested exist
            for name in requested_experiments.split(','):
                if name not in existing_names:
                    raise DEVICE_EXCEPTION(f"YOU CALLED FOR '{name}', BUT NOBODY CAME")
        
        #Return the intersection of requested names and existing names
        return [name for name in requested_experiments.split(",") if name in existing_names]
    
    #----------------------------------------------------
    
    def list_experiments(self) -> list[str]:
        return [name for name in self.database.list_collection_names() if name != "temp"]
    
    #----------------------------------------------------
    
    def is_temp(self) -> bool:
        current_collection = self.sub_path
        self.sub_path = "temp"
        temp_dict = self.collection.find_one({current_collection: 1})
        self.sub_path = current_collection
        return temp_dict is not None
    
    #----------------------------------------------------

    def set_temp(self):
        current_collection = self.sub_path
        self.sub_path = "temp"

        #dummy document
        self.collection.update_one(
            { current_collection: 1 },
            { '$setOnInsert': { current_collection: 1 }},
            upsert=True
        )

        self.sub_path = current_collection

#==========================================================================================================