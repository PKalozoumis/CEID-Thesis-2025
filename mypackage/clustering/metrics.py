import numpy as np
from hdbscan.validity import validity_index
from itertools import chain
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import davies_bouldin_score

from rich.console import Console
from rich.table import Table

from .classes import ChainCluster, ChainClustering
from ..helper import create_table

#================================================================================================

def within_cluster_similarity(cluster: ChainCluster, flat: bool = False) -> float:
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
    
    if flat:
        mat = cluster.sentence_matrix()
    else:
        mat = cluster.chain_matrix()
    
    sim = cosine_similarity(mat, mat)
    res = (np.sum(sim, axis=1) - 1) / (len(cluster) - 1)
    return np.average(res)

#================================================================================================

def avg_within_cluster_similarity(clustering: ChainClustering, flat: bool = False):
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

    vec = np.array([within_cluster_similarity(cluster, flat=flat) for key, cluster in clustering.clusters.items() if key >= 0])
    if len(vec) > 0:
        return np.average(vec)
    else:
        return -1
    
#================================================================================================

def cluster_centroid_similarity(cluster: ChainCluster, *, allow_self_similarity: bool = False, flat: bool = False):
    if len(cluster) == 1:
        return 1

    if flat:
        mat = cluster.sentence_matrix()
    else:
        mat = cluster.chain_matrix()
    
    sim = cosine_similarity(mat, cluster.vector.reshape((1,-1)))

    if not allow_self_similarity and cluster.pooling_method in ChainCluster.EXEMPLAR_BASED_METHODS:
        return (np.sum(sim) - 1) / (len(cluster) - 1)
    else:
        return np.average(sim)
    
#================================================================================================

def avg_cluster_centroid_similarity(clustering: ChainClustering, *, allow_self_similarity: bool = False, flat: bool = False):
    vec = np.array([cluster_centroid_similarity(cluster, allow_self_similarity=allow_self_similarity, flat=flat) for cluster in clustering if cluster.label > -1])
    if len(vec) > 0:
        return np.average(vec)
    else:
        return -1
    
#================================================================================================

def chain_clustering_silhouette_score(clustering: ChainClustering, reducer=None, flat: bool = False):

    #Filter out outliers
    chains = [chain for chain, label in zip(clustering.chains, clustering.labels) if label >= 0]
    labels = [label for label in clustering.labels if label >= 0]

    if len(chains) == 0:
        return None #How did we get here
    
    if flat:
        labels, sentences = zip(*[(label, sentence) for label, chain in zip(labels, chains) for sentence in chain])
        mat = np.array([s.vector for s in sentences])
    else:
        mat = np.array([chain.vector for chain in chains])

    if reducer:
        mat = reducer.fit_transform(mat)
        return silhouette_score(mat, labels, metric='euclidean')
    else:
        return silhouette_score(mat, labels, metric='cosine')

#================================================================================================

def dbcv(clustering: ChainClustering, reducer = None):
    if reducer:
        mat = reducer.fit_transform(np.array([c.vector for c in clustering.chains])).astype(np.float64)
    else:
        mat = np.array([c.vector for c in clustering.chains])
    return validity_index(mat, np.array(clustering.labels))

def dbi(clustering: ChainClustering, reducer = None):
    if reducer:
        mat = reducer.fit_transform(np.array([c.vector for c in clustering.chains])).astype(np.float64)
    else:
        mat = np.array([c.vector for c in clustering.chains])
    return davies_bouldin_score(mat, clustering.labels)

#================================================================================================

VALID_METRICS = ["silhouette", "flat_silhouette"]

def clustering_metrics(clustering: ChainClustering, metrics_list: list[str] = None, *, reducer=None, value=False, render=False, return_renderable=False) -> dict | tuple[dict, Table]:
    metrics = {
        'silhouette': {'name': "Silhouette Score", 'value': lambda: chain_clustering_silhouette_score(clustering, reducer)},
        'flat_silhouette': {'name': "Flat Silhouette Score", 'value': lambda: chain_clustering_silhouette_score(clustering, reducer, flat=True)},
        'avg_sim': {'name': "Average Within-Cluster Similarity", 'value': lambda: avg_within_cluster_similarity(clustering)},
        'dbcv': {'name': "DBCV", 'value': lambda: dbcv(clustering, reducer)},
        'avg_centroid_sim': {'name': "Average Similarity to Centroid", 'value': lambda: avg_cluster_centroid_similarity(clustering, allow_self_similarity=True)},
        'avg_flat_centroid_sim': {'name': "Average Similarity to Centroid", 'value': lambda: avg_cluster_centroid_similarity(clustering, allow_self_similarity=True, flat=True)},
        'avg_flat_sim': {'name': "Average Within-Cluster Similarity", 'value': lambda: avg_within_cluster_similarity(clustering, flat=True)},
        'avg_flat_cluster_size': {'name': 'Flat Cluster Size', 'value': lambda: np.average([len(cl.sentence_matrix()) for cl in clustering.clusters.values()])}
    }

    #Run requested results
    for k in metrics:
        if metrics_list is None or k in metrics_list:
            metrics[k]['value'] = metrics[k]['value']() #Run the lazy function
        else:
            metrics[k]['value'] = None

    #Delete None
    metrics = {k:v for k,v in metrics.items() if v['value'] is not None}

    #Keep only value
    if value:
        for k in metrics:
            metrics[k] = metrics[k]['value']

    if len(metrics) == 0:
        metrics = {}
    '''
    elif len(metrics) == 1:
        metrics = list(metrics.values())[0]
    '''

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