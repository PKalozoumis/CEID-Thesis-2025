from .clustering import ChainCluster
from .elastic import Session, Document, ElasticDocument
from .sentence import Sentence, SentenceChain, SentenceLike, split_to_sentences
import pickle
import os
import sys
from dataclasses import dataclass, field

@dataclass
class ProcessedDocument():
    doc: Document
    chains: list[SentenceChain]
    labels: list[int]
    clusters: dict[int, ChainCluster]
    params: dict = field(default=None)

#Things relating to the document itself will be retrieved either from elasticsearch or from the cache
#   Document has already been retrieved. Also, we do not care about its type. Could be Document or ElasticDocument
#We should not store text, filter_path/text_path/etc or the sentences themselves
#We do store the id, so that we can point back to the document
#The sentences will be broken down upon retrieval
#...but we do need to store their offsets and embeddings
#...as well as the chains that have formed (with their representative, and all other information)
#Same thing with the chain clusters: we need to store everything

def save_clusters(clusters: dict, path: str, *, params: dict = None):
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
    out = [c.data() for c in clusters.values()]

    with open(os.path.join(path, f"{out[0]['id']}.pkl"), "wb") as f:
        pickle.dump({'params': params, 'data': out}, f)

#=====================================================================================================

def restore_clusters(doc: Document, path: str) -> ProcessedDocument:
    '''
    Restores all cluster objects for a specific document from pickle files
    '''
    with open(os.path.join(path, f"{doc.id}.pkl"), "rb") as f:
        data = pickle.load(f)

    if type(data) is list:
        params = None
    else:
        params = data['params']
        data = data['data']

    #Recreate the cluster dictionary, by mapping each label to its ChainCluster,
    #the same way the clusters are returned from chain_clustering
    clusters = {}

    #From our cluster objects, we want to get back the labels for all the chains
    offset_and_label = []

    out = []

    for cluster_data in data:
        #Recreate this cluster from its data
        cluster = ChainCluster.from_data(cluster_data, doc)
        clusters[cluster.label] = cluster
        offset_and_label.extend((chain, cluster.label) for chain in cluster)
    
    offset_and_label.sort(key=lambda tup: tup[0].offset)
    chains = list([tup[0] for tup in offset_and_label])
    labels = list([tup[1] for tup in offset_and_label])

    return ProcessedDocument(doc, chains, labels, clusters, params)

#=====================================================================================================

def load_pickles(sess: Session, path: str, docs: int|list[int]) -> ProcessedDocument|list[ProcessedDocument]:
    out = []

    if type(docs) is int:
        temp = [docs]
    else:
        temp = docs

    for id in temp:
        out.append(restore_clusters(ElasticDocument(sess, id, text_path="article"), path))

    if type(docs) is int:
        return out[0]
    else:
        return out