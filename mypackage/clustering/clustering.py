from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from ..sentence import SentenceChain
from .classes import ChainCluster
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from umap import UMAP
from numpy import ndarray
import warnings

#===================================================================================================

def group_chains_by_label(chains: list[SentenceChain], clustering: list[int]) -> dict[int, list[SentenceChain]]:
    '''
    Groups the chains into lists based on the labels returned by ```chain_clustering```

    Args:
        chains (list[SentenceChain]): The original set of chains
        clustering (list[int]): A list of cluster labels. One label for each chain in ```chains```. This is the result of ```chain_clustering```

    Returns:
        dict[int[SentenceChain]]: A dictionary of clusters. Each cluster is a list of chains
    '''

    clusters = {}
    
    for chain, label in zip(chains, clustering):
        if label in clusters:
            clusters[label].append(chain)
        else:
            clusters[label] = [chain]

    return clusters

#===================================================================================================

def label_positions(labels: list[int]) -> dict[int, list[int]]:
    '''
    Inverts the label list. For each label, it returns the indices where it occurs

    Args:
        clustering (list[int]): A list of cluster labels. One label for each chain in ```chains```. This is the result of ```chain_clustering```

    Returns:
        dict[int[int]]: A dictionary mapping each label to its indices
    '''

    indices = {}
    
    for i, label in enumerate(labels):
        if label in indices:
            indices[label].append(i)
        else:
            indices[label] = [i]

    return indices

#===================================================================================================

def chain_clustering(chains: list[SentenceChain]) -> tuple[list[int], dict[int, ChainCluster]]:
    '''
    Clusters a list of sentence chains for a single document

    Parameters
    --------------------------------------------------------
    chain: list[SentenceChain]
        The list of chains to cluster

    Returns
    --------------------------------------------------------
    labels: list[int]
        A list of labels. One label for each input chain

    clustered_chains: dict[int, ChainCluster]
        A dictionary of clusters, with the label as the key
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) #when using seed
        warnings.filterwarnings("ignore", category=FutureWarning) #not in my control

        #Extract representative vectors from the chains
        #Set them as rows of a new matrix
        matrix = np.array([chain.vector for chain in chains])

        #Reduce dimensionality before clustering
        clustering_reducer = UMAP(n_components=10, metric="cosine", random_state=42)
        reduced_matrix = clustering_reducer.fit_transform(matrix)

        #Cluster
        model = HDBSCAN(min_cluster_size=3, min_samples=5,metric="cosine")
        clustering = model.fit(reduced_matrix)

    clusters = group_chains_by_label(chains, clustering.labels_)
    cluster_objects = {}

    #Create cluster objects
    for label, cluster in clusters.items():
        cluster_objects[label] = ChainCluster(cluster, label)

    return list(clustering.labels_), cluster_objects

#===================================================================================================

def visualize_clustering(chains: list[SentenceChain], clustering_labels: list[int]):
    '''
    Visualize clustered chains

    Args:
        chains (list[SentenceChain]): The original set of chains
        clustering (list[int]): A list of cluster labels. One label for each chain in ```chains```. This is the result of ```chain_clustering```
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) #when using seed

        matrix = np.array([chain.vector for chain in chains])
        
        colors = [sns.color_palette()[label] if label >= 0 else (0,0,0) for label in clustering_labels]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            visualization_reducer = UMAP(n_components=10, metric="cosine", random_state=42)
        
        reduced = visualization_reducer.fit_transform(matrix)

        plt.scatter(reduced[:, 0], reduced[:, 1], c=colors)
        plt.show()

#===================================================================================================
