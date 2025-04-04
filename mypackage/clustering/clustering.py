from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from ..sentence import SentenceChain
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from umap import UMAP
from numpy import ndarray

def chain_clustering(chains: list[SentenceChain]) -> ndarray:
    '''
    Clusters a list of sentence chains for a single document

    Args:
        chains (list[SentenceChain): The list of chains to clusters

    Returns:
        
    '''
    #Extract representative vectors from the chains
    #Set them as rows of a new matrix
    matrix = np.array([chain.vector for chain in chains])

    #Reduce dimensionality before clustering
    clustering_reducer = UMAP(n_components=10, metric="cosine", random_state=42)
    reduced_matrix = clustering_reducer.fit_transform(matrix)

    #Cluster
    model = HDBSCAN(min_cluster_size=3, min_samples=5,metric="cosine", store_centers="medoid")
    clustering = model.fit(reduced_matrix)
    return list(clustering.labels_)


def visualize_clustering(chains: list[SentenceChain], clustering: list[int]) -> None:
    '''
    '''
    matrix = np.array([chain.vector for chain in chains])
    
    colors = [sns.color_palette()[label] if label >= 0 else (0,0,0) for label in clustering.labels_]
    visualization_reducer = UMAP(n_components=2, metric="cosine", random_state=42)
    reduced = visualization_reducer.fit_transform(matrix)

    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors)
    plt.show()