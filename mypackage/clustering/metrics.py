
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.metrics import davies_bouldin_score
from ..sentence import SentenceChain
from .classes import ChainCluster, ChainClustering
from ..helper import create_table
from .clustering import group_chains_by_label
import numpy as np
from rich.console import Console
from rich.table import Table
from hdbscan.validity import validity_index
from dbcv import dbcv
from itertools import chain

#================================================================================================

def within_cluster_similarity(cluster: ChainCluster) -> float:
    '''
    Caclculates the average similarity of every pair of sentences in the cluster. 

    Arguments
    ---
    chains: list[SentenceChain]
        The chains in the cluster to calculate similarity for
    
    Returns
    ---
    sim: float
        Average similarity within the cluster
    '''
    if len(cluster) == 1:
        return 1
    
    mat = cluster.chain_matrix()
    sim = cosine_similarity(mat, mat)
    res = (np.sum(sim, axis=1) - 1) / (len(cluster) - 1)
    return np.average(res)

#================================================================================================

def avg_within_cluster_similarity(clustering: ChainClustering):
    '''
    Caclculates the average similarity within each cluster in the list.
    Then, it calculates the average of those values. 

    Arguments
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

    vec = np.array([within_cluster_similarity(cluster) for key, cluster in clustering.clusters.items() if key >= 0])
    if len(vec) > 0:
        return np.average(vec)
    else:
        return -1
    
#================================================================================================

def cluster_centroid_similarity(cluster: ChainCluster):
    if len(cluster) == 1:
        return 1

    mat = cluster.chain_matrix()
    sim = cosine_similarity(mat, cluster.vector.reshape((1,-1)))

    if cluster.pooling_method in ChainCluster.EXEMPLAR_BASED_METHODS:
        return (np.sum(sim) - 1) / (len(cluster) - 1)
    else:
        return np.average(sim)
    
#================================================================================================

def avg_cluster_centroid_similarity(clustering: ChainClustering):
    vec = np.array([cluster_centroid_similarity(cluster) for cluster in clustering if cluster.label > -1])
    if len(vec) > 0:
        return np.average(vec)
    else:
        return -1
    
#================================================================================================

def chain_clustering_silhouette_score(clustering: ChainClustering):

    #Filter out outliers
    chains = [chain for chain, label in zip(clustering.chains, clustering.labels) if label >= 0]
    labels = [label for label in clustering.labels if label >= 0]

    if len(chains) == 0:
        return None #How did we get here

    #From each chain in the list, get its representative vector
    #Crete a matrix from these vectors
    mat = np.array([chain.vector for chain in chains])
    return silhouette_score(chains, labels, metric='cosine')

#================================================================================================

def chain_clustering_flat_silhouette_score(clustering: ChainClustering):

    #Filter out outliers
    chains = [chain for chain, label in zip(clustering.chains, clustering.labels) if label >= 0]
    labels = [label for label in clustering.labels if label >= 0]

    if len(chains) == 0:
        return None

    #We need to expand each chain to its sentences
    labels, sentences = zip(*[(label, sentence) for label, chain in zip(labels, chains) for sentence in chain])

    #From each sentence in the list, get its representative vector
    #Crete a matrix from these vectors
    return silhouette_score(sentences, labels, metric='cosine')

#================================================================================================

VALID_METRICS = ["silhouette", "flat_silhouette"]

def clustering_metrics(clustering: ChainClustering, *, render=False, return_renderable=False) -> dict | tuple[dict, Table]:

    metrics = {
        'silhouette': {'name': "Silhouette Score", 'value': chain_clustering_silhouette_score(clustering)},
        'flat_silhouette': {'name': "Flat Silhouette Score", 'value': chain_clustering_flat_silhouette_score(clustering)},
        'avg_sim': {'name': "Average Within-Cluster Similarity", 'value': avg_within_cluster_similarity(clustering)},
        #'validity': {'name': "Validity", 'value': validity_index(distas, np.array(labels), metric="precomputed", d=chains[0].vector.shape[0])},
        #'dbcv': {'name': "DBCV", 'value': dbcv(chains, labels)},
        'dbi': {'name': "Davies-Bouldin Index", 'value': davies_bouldin_score(clustering.chains, clustering.labels)},
        'avg_centroid_sim': {'name': "Average Similarity to Centroid", 'value': avg_cluster_centroid_similarity(clustering)},
    }

    console = Console()
    
    if render or return_renderable:
        table = create_table(['Metric', 'Score'], {temp['name']:temp['value'] for temp in metrics.values()}, title="Clustering Metrics")
        if render:
            console.print(table)
        if return_renderable:
            return metrics, table 
        
    return metrics

#=================================================================================================================

def stats(clustering: ChainClustering):
    chain_lengths = [len(c) for c in clustering.chains]

    data = {}

    data['num_chains'] = len(clustering.chains)
    data['avg_chain_length'] = np.average(chain_lengths)
    data['min_chain_length'] = np.min(chain_lengths)
    data['max_chain_length'] = np.max(chain_lengths)

    sentence_lengths = [len(c) for c in chain.from_iterable(clustering.chains)]
    data['num_sentences'] = len(sentence_lengths)
    data['num_words'] = np.sum(sentence_lengths)
    data['avg_sentence_length'] = np.average(sentence_lengths)
    data['min_sentence_length'] = np.min(sentence_lengths)
    data['max_sentence_length'] = np.max(sentence_lengths)

    data['num_clusters'] = len(clustering.clusters) - (1 if -1 in clustering.clusters else 0)

    return data

#=================================================================================================================

def cluster_stats(cluster: ChainCluster):
    chain_lengths = [len(c) for c in cluster.chains]

    data = {}

    data['num_chains'] = len(cluster.chains)
    data['avg_chain_length'] = np.average(chain_lengths)
    data['min_chain_length'] = np.min(chain_lengths)
    data['max_chain_length'] = np.max(chain_lengths)

    sentence_lengths = [len(c) for c in chain.from_iterable(cluster.chains)]
    data['num_sentences'] = len(sentence_lengths)
    data['num_words'] = np.sum(sentence_lengths)
    data['avg_sentence_length'] = np.average(sentence_lengths)
    data['min_sentence_length'] = np.min(sentence_lengths)
    data['max_sentence_length'] = np.max(sentence_lengths)

    return data