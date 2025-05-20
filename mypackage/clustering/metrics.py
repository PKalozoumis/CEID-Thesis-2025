
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.metrics import davies_bouldin_score
from ..sentence import SentenceChain
from .classes import ChainCluster
from ..helper import create_table
from .clustering import group_chains_by_label
import numpy as np
from rich.console import Console
from rich.table import Table
from hdbscan.validity import validity_index
from dbcv import dbcv

#================================================================================================

def within_cluster_similarity(chains: list[SentenceChain]) -> float:
    '''
    Caclculates the average similarity of every pair of sentences in the cluster. 

    Args
    ---
    chains: list[SentenceChain]
        The chains in the cluster to calculate similarity for
    
    Returns
    ---
    sim: float
        Average similarity within the cluster
    '''
    if len(chains) == 1:
        return 1
    
    mat = [chain.vector for chain in chains]
    sim = cosine_similarity(mat, mat)
    res = (np.sum(sim, axis=1) - 1) / (len(chains) - 1)
    return np.average(res)

#================================================================================================

def avg_within_cluster_similarity(chains: list[SentenceChain], labels: list[int]):
    '''
    Caclculates the average similarity within each cluster in the list.
    Then, it calculates the average of those values. 

    Args
    ---
    chains: list[SentenceChain]
        The full list of chains across all clusters

    labels: list[int]
        The cluster labels for each chain in ```chains```
    
    Returns
    ---
    avg_sim: float
        Average within-cluster similarity
    '''
    clusters = group_chains_by_label(chains, labels)

    vec = np.array([within_cluster_similarity(cluster_chains) for key, cluster_chains in clusters.items() if key >= 0])
    if len(vec) > 0:
        return np.average(vec)
    else:
        return -1
    
#================================================================================================

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
        'avg_sim': {'name': "Average Within-Cluster Similarity", 'value': avg_within_cluster_similarity(chains, labels)},
        #'validity': {'name': "Validity", 'value': validity_index(distas, np.array(labels), metric="precomputed", d=chains[0].vector.shape[0])},
        #'dbcv': {'name': "DBCV", 'value': dbcv(chains, labels)},
        'dbi': {'name': "Davies-Bouldin Index", 'value': davies_bouldin_score(chains, labels)}
    }

    console = Console()
    
    if render or return_renderable:
        table = create_table(['Metric', 'Score'], {temp['name']:temp['value'] for temp in metrics.values()}, title="Clustering Metrics")
        if render:
            console.print(table)
        if return_renderable:
            return metrics, table 
        
    return metrics