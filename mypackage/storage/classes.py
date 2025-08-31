from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, overload
from more_itertools import always_iterable
import os
import pickle
import shutil
from pymongo import MongoClient, collection, database
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from collections import defaultdict
import re
from types import SimpleNamespace

from ..clustering import ChainCluster, ChainClustering
from ..elastic import Document, Session, ElasticDocument
from ..sentence import Sentence, SentenceChain
from ..helper import DEVICE_EXCEPTION
from ..summarization import SummaryUnit
from ..query import Query
from ..cluster_selection import SelectedCluster, RelevanceEvaluator
from ..experiments import ExperimentManager

from elasticsearch import NotFoundError

from dataclasses import dataclass

#==========================================================================================================

@dataclass
class RealTimeResults():
    '''
    Represents the results of a single client query. Can be used for later evaluatiion.
    '''
    #Environment objects
    sess: Session
    experiment: str
    db: DatabaseSession
    evaluator: RelevanceEvaluator
    query: Query
    args: SimpleNamespace

    #Result objects
    returned_docs: list[ElasticDocument]
    selected_clusters: list[SelectedCluster]
    summaries: list[SummaryUnit]
    times: list[dict] #We can gather the times from multiple runs (by merging different files)

    #---------------------------------------------------------------------------------

    @staticmethod
    def store_results(
        path: str,

        sess: Session,
        query: Query,
        evaluator: RelevanceEvaluator,
        args: SimpleNamespace,

        returned_docs: list[ElasticDocument],
        selected_clusters: list[SelectedCluster],
        summaries: list[SummaryUnit],

        times: defaultdict
    ):
        data = {
            'index_name': sess.index_name,
            'experiment': args.experiment,
            'query_id': query.id,
            'cross_encoder': evaluator.model_name,
            'args': args,

            'returned_docs': [doc.id for doc in returned_docs],
            'selected_clusters': [c.data() for c in selected_clusters],
            'summaries': [s.data() for s in summaries],

            'times': [dict(times)]
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    #---------------------------------------------------------------------------------

    @classmethod
    def load(cls, path: str, sess: Session, db: DatabaseSession, exp_manager: ExperimentManager):
        #I want to load a results file that corresponds to a specific:
            #Index
            #Experiment
            #Query

        #What do we NOT know about the environment:
            #The index name
            #The experiment name
            #We don't care about the database
            #The query ID
            #The cross-encoder used
            #The client arguments (explicitly write ALL, even the default ones)

        #Result objects that need to be stored:
            #Retrieved documents
            #Selected clusters
            #Generated summaries

        with open(path, "rb") as f:
            data = pickle.load(f)

        #Check if the index name matches
        if data['index_name'] != sess.index_name:
            raise Exception("Index names do not match")
        
        #Load experiment
        db.sub_path = data['args'].experiment
        processed = db.load(sess, data['returned_docs'], skip_missing_docs=False)
        
        query = exp_manager.get_queries(data['query_id'], data['index_name'])[0]
        evaluator = RelevanceEvaluator(query, data['cross_encoder'])

        #Find the document (inside returned docs) that corresponds to each selected cluster
        #This will give the position of the respective processed document
        selected_clusters = [
            SelectedCluster.from_data(
                sc_data,
                processed[data['returned_docs'].index(sc_data['doc_id'])].clustering,
                evaluator
            )
            for sc_data in data['selected_clusters']
        ]

        #Documents
        returned_docs = [
            ElasticDocument(
                sess,
                id,
                text_path = exp_manager.index_defaults[sess.index_name]['text_path']
            )
            for id in data['returned_docs']
        ]
        
        #Create final object
        return cls(
            sess,
            db.sub_path,
            db,
            evaluator,
            query,
            data['args'],

            returned_docs,            
            selected_clusters,
            [SummaryUnit.from_data(summ, selected_clusters) for summ in data['summaries']],

            data['times']      
        )

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

    @property
    @abstractmethod
    def base_path(self) -> str:
        pass

    @property
    @abstractmethod
    def sub_path(self) -> str:
        pass

    @base_path.setter
    @abstractmethod
    def base_path(self, value: str):
        pass
    
    @sub_path.setter
    @abstractmethod
    def sub_path(self, value: str):
        pass

    @classmethod
    @abstractmethod
    def duplicate_session(cls, db: DatabaseSession) -> DatabaseSession: ...

    @overload
    def load(self, sess: Session, docs: int|ElasticDocument, *, skip_missing_docs: bool = False) -> ProcessedDocument: ...

    @overload
    def load(self, sess: Session, docs: list[int]|list[ElasticDocument], *, skip_missing_docs: bool = False) -> list[ProcessedDocument]: ...

    @abstractmethod
    def load(self, sess: Session, docs: int|list[int]|ElasticDocument|list[ElasticDocument], *, skip_missing_docs: bool = False) -> ProcessedDocument|list[ProcessedDocument]:
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
    def get_experiment_params(self) -> dict:
        pass

    @abstractmethod
    def is_temp(self) -> bool:
        pass

    @abstractmethod
    def set_temp(self):
        pass

    def _restore_clusters(self, doc: Document, data: dict, params: dict = None, skip_missing_docs: bool = False) -> ProcessedDocument:
        '''
        Recreates all cluster objects for a specific document using the retrieved data
        '''
        #Ensure the document's text has been retrieved
        #Otherwise, we cannot get the sentences
        #May throw NotFoundError exception if document does not exist
        try:
            doc.get()
        except NotFoundError as e:
            if skip_missing_docs:
                return None
            else:
                raise e

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

    @classmethod
    def duplicate_session(cls, db: PickleSession) -> PickleSession:
        return cls(db._base_path, db._sub_path)

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
    def load(self, sess: Session, docs: int|ElasticDocument, *, skip_missing_docs: bool = False) -> ProcessedDocument: ...

    @overload
    def load(self, sess: Session, docs: list[int]|list[ElasticDocument], *, skip_missing_docs: bool = False) -> list[ProcessedDocument]: ...

    def load(self, sess: Session, docs: int|list[int]|ElasticDocument|list[ElasticDocument], *, skip_missing_docs: bool = False) -> ProcessedDocument|list[ProcessedDocument]:
        out = []

        for doc in always_iterable(docs):

            if doc is None:
                out.append(None)
                continue
            
            #We only passed a document id, so we have to retrieve it first
            if type(doc) is int:
                doc_obj = ElasticDocument(sess, doc, text_path="article")
            #We passed an existing document, so we use it
            else:
                doc_obj = doc

            #Retrieve pickle
            with open(os.path.join(self.full_path, f"{doc_obj.id}.pkl"), "rb") as f:
                data = pickle.load(f)

            if 'params' in data:
                params = data['params']
                data = data['data']
            else:
                params = None

            out.append(self._restore_clusters(doc_obj, data, params, skip_missing_docs=skip_missing_docs))

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

    def get_experiment_params(self) -> dict:
        found = None
        for filename in os.listdir(self.full_path):
            if filename.endswith(".pkl"):
                found = os.path.join(self.full_path, filename)

        with open(found, "rb") as f:
            data = pickle.load(f)

        if 'params' in data:
            return data['params']
        else:
            return None
    
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

    def __init__(self, *, host: str = "localhost:27017", db_name: str | None = None, collection: str | None = None):
        self.host = host
        self._base_path = db_name
        self._sub_path = collection

        max_retries = 5
        for attempt in range(0, max_retries + 2):
            try:
                self.client = MongoClient(f"mongodb://{host}/", serverSelectionTimeoutMS=7000)
                self.client.admin.command("ping")
            except (ServerSelectionTimeoutError, ConnectionFailure) as e:
                if attempt < max_retries:
                    print(f"Could not connect to MongoDB database. Retrying... ({attempt+1}/{max_retries})")
                else:
                    raise e
                
    #----------------------------------------------------

    @property
    def db_type(self) -> str:
        return "mongo"
    
    #----------------------------------------------------

    @classmethod
    def duplicate_session(cls, db: MongoSession) -> MongoSession:
        return cls(host=db.host, db_name=db._base_path, collection=db._sub_path)

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
    def load(self, sess: Session, docs: int|ElasticDocument, *, skip_missing_docs: bool = False) -> ProcessedDocument: ...

    @overload
    def load(self, sess: Session, docs: list[int]|list[ElasticDocument], *, skip_missing_docs: bool = False) -> list[ProcessedDocument]: ...

    def load(self, sess: Session, docs: int|list[int]|ElasticDocument|list[ElasticDocument], *, skip_missing_docs: bool = False) -> ProcessedDocument|list[ProcessedDocument]:
        out = []

        doc_objects = []
        doc_ids = []

        #Get document objects from input
        for doc in always_iterable(docs):
            if doc is None:
                out.append(None)
                continue

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

        '''
        #Check if we have retrieved data for all docs
        for doc_id in doc_ids:
            if not doc_to_clusters[doc_id]:
                raise Exception(f"Preprocessing results for document {doc_id} not found")
        '''

        #Retrieve params
        params = self.database['metadata'].find_one({'collection': self.sub_path})['params']
    
        #Create clusters
        for doc_obj in doc_objects:
           out.append(self._restore_clusters(doc_obj, doc_to_clusters[doc_obj.id], params, skip_missing_docs=skip_missing_docs))

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

        #Insert metadata
        self.database['metadata'].update_one(
            {"collection": self.sub_path},
            {"$set": {"params": params}},
            upsert=True
        )

    #----------------------------------------------------

    def close(self):
        self.client.close()

    #----------------------------------------------------

    def delete(self):
        self.collection.drop()
        #Delete the respective entry in the temp collection, if exists
        self.database['metadata'].delete_many({'collection': self.sub_path})

    #----------------------------------------------------

    def available_experiments(self, requested_experiments: str) -> list[str]:
        '''
        Returns the names of the available experiments
        '''
        existing_names = self.database.list_collection_names()
        if 'metadata' in existing_names:
            existing_names.remove('metadata')

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
        return [name for name in self.database.list_collection_names() if name != "metadata"]
    
    #----------------------------------------------------

    def get_experiment_params(self) -> dict:
        metadata_dict = self.database['metadata'].find_one({"collection": self.sub_path})
        return metadata_dict['params']

    #----------------------------------------------------
    
    def is_temp(self) -> bool:
        current_collection = self.sub_path
        self.sub_path = "metadata"
        metadata_dict = self.collection.find_one({"collection": current_collection})
        self.sub_path = current_collection
        return metadata_dict is not None and metadata_dict.get('temp', False)
    
    #----------------------------------------------------

    def set_temp(self):
        current_collection = self.sub_path
        self.sub_path = "metadata"

        self.collection.update_one(
            {"collection": current_collection},
            {"$set": {"temp": True}},
            upsert=True
        )

        self.sub_path = current_collection

#==========================================================================================================