
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from ..sentence import SentenceChain
from .classes import ChainCluster
from ..helper import create_table
import numpy as np
from rich.console import Console
from rich.table import Table
from hdbscan.validity import validity_index
from dbcv import dbcv

def chain_clustering_silhouette_score(chains: list[SentenceChain], labels: list[int]):

    #Filter out outliers
    chains = [chain for chain, label in zip(chains, labels) if label >= 0]
    labels = [label for label in labels if label >= 0]

    if len(chains) == 0:
        return None #How did we get here

    #From each chain in the list, get its representative vector
    #Crete a matrix from these vectors
    mat = np.array([chain.vector for chain in chains])
    return silhouette_score(chains, labels, metric='cosine')

#================================================================================================

def chain_clustering_flat_silhouette_score(chains: list[SentenceChain], labels: list[int]):

    #Filter out outliers
    chains = [chain for chain, label in zip(chains, labels) if label >= 0]
    labels = [label for label in labels if label >= 0]

    if len(chains) == 0:
        return None

    #We need to expand each chain to its sentences
    labels, sentences = zip(*[(label, sentence) for label, chain in zip(labels, chains) for sentence in chain])

    #From each sentence in the list, get its representative vector
    #Crete a matrix from these vectors
    return silhouette_score(sentences, labels, metric='cosine')

#================================================================================================

VALID_METRICS = ["silhouette", "flat_silhouette"]

def clustering_metrics(chains: list[SentenceChain], labels: list[int], *, render=False, return_renderable=False) -> dict | tuple[dict, Table]:

    '''
    def find_duplicate_chains(chains):
        console = Console()
        arr = [chain.vector for chain in chains]
        _, counts = np.unique(arr, axis=0, return_counts=True)
        duplicates = np.unique(arr, axis=0)[counts > 1]
        console.print([chain.text for chain in chains if any(np.array_equal(chain.vector, d) for d in duplicates)])
        #print(duplicates[0][duplicates[1] > 1])

    find_duplicate_chains(chains)
    '''

    distas = cosine_distances(np.array([chain.vector for chain in chains])).astype(np.float64)

    metrics = {
        'silhouette': {'name': "Silhouette Score", 'value': chain_clustering_silhouette_score(chains, labels)},
        'flat_silhouette': {'name': "Flat Silhouette Score", 'value': chain_clustering_flat_silhouette_score(chains, labels)},
        #'validity': {'name': "Validity", 'value': validity_index(distas, np.array(labels), metric="precomputed", d=chains[0].vector.shape[0])},
        'dbcv': {'name': "DBCV", 'value': dbcv(chains, labels)}
    }

    console = Console()
    table = create_table(['Metric', 'Score'], {temp['name']:temp['value'] for temp in metrics.values()}, title="Clustering Metrics")
    
    if render:
        console.print(table)

    if return_renderable:
        return metrics, table 
    
    return metrics