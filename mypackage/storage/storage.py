from ..clustering import ChainCluster, ChainClustering
from ..elastic import Session, Document, ElasticDocument
from ..sentence import SentenceChain
from .classes import ProcessedDocument
from ..cluster_selection import SelectedCluster
import pickle
import os
from typing import overload
import json

#==========================================================================================================

def save_clusters(clustering: ChainClustering, path: str, *, params: dict = None):
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
    offset_and_label: list[tuple[SentenceChain, int]]

    out = []

    for cluster_data in data:
        #Recreate this cluster from its data
        cluster = ChainCluster.from_data(cluster_data, doc)
        clusters[cluster.label] = cluster
        offset_and_label.extend((chain, cluster.label) for chain in cluster)
    
    offset_and_label.sort(key=lambda tup: tup[0].offset)
    chains = list([tup[0] for tup in offset_and_label])
    labels = list([tup[1] for tup in offset_and_label])

    #Assign index to each chain
    #(temporary, because currently the index is not stored in the pickles)
    for i, chain in enumerate(chains):
        chain.index = i

    sentences = [sentence for chain in chains for sentence in chain]

    return ProcessedDocument(doc, ChainClustering(chains, labels, clusters), sentences, params)

#=====================================================================================================

@overload
def load_pickles(sess: Session, path: str, docs: int|ElasticDocument) -> ProcessedDocument: ...

@overload
def load_pickles(sess: Session, path: str, docs: list[int]|list[ElasticDocument]) -> list[ProcessedDocument]: ...

def load_pickles(sess: Session, path: str, docs: int|list[int]|ElasticDocument|list[ElasticDocument]) -> ProcessedDocument|list[ProcessedDocument]:
    out = []

    if isinstance(docs, list):
        temp = docs
    else:
        temp = [docs]

    for item in temp:
        if isinstance(item, ElasticDocument):
            #We passed an existing document, so we use it
            out.append(restore_clusters(item, path))
        elif type(item) is int:
            #We only passed a document id, so we have to retrieve it first
            out.append(restore_clusters(ElasticDocument(sess, item, text_path="article"), path))

    if isinstance(docs, list):
        return out
    else:
        return out[0]
    
#=============================================================================================================

def store_summary(cluster: SelectedCluster, summary: str, args):
    os.makedirs("generated_summaries", exist_ok=True)

    with open(f"generated_summaries/cluster_{cluster.id}.json", "w") as f:
        json.dump({
            'generated_with': "llm" if args.m == "llm" else "transformer",
            'summary': summary
        }, f)

#==============================================================================================================

def load_summary(cluster: SelectedCluster):
    if not os.path.exists(f"generated_summaries/cluster_{cluster.id}.json"):
        return None

    with open(f"generated_summaries/cluster_{cluster.id}.json", "r") as f:
        return json.load(f)['summary']
