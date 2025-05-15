from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from ..sentence import SentenceChain
from .classes import ChainCluster
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from umap import UMAP
from numpy import ndarray
import warnings
from matplotlib.patches import Patch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#===================================================================================================

def group_chains_by_label(chains: list[SentenceChain], clustering: list[int]) -> dict[int, list[SentenceChain]]:
    '''
    Groups the chains into lists based on the labels returned by ```chain_clustering```.
    The chains are ordered 

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

def chain_clustering(chains: list[SentenceChain], n_components: int = 25, min_dista: float = 0.1) -> tuple[list[int], dict[int, ChainCluster]]:
    '''
    Clusters a list of sentence chains for a single document.
    The chains inside each returned cluster are ordered based on their offset inside the document

    Arguments
    --------------------------------------------------------
    chain: list[SentenceChain]
        The list of chains to cluster

    n_components: int
        The number of dimensions to reduce the embedding space to.
        Set to ```None``` to skip dimensionality reduction

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

        if n_components is not None:
            #Reduce dimensionality before clustering
            clustering_reducer = UMAP(n_components=n_components, metric="cosine", random_state=42, min_dist=min_dista)
            reduced_matrix = clustering_reducer.fit_transform(matrix)
        else:
            reduced_matrix = matrix

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

def visualize_clustering(chains: list[SentenceChain], clustering_labels: list[int],*,save_to: str | None = None, show: bool = False, ax: Axes = None, return_legend: bool = False, min_dista: float = 0.1):
    '''
    Creates a scatter plot of the clustered chains

    Arguments
    ---
    chains: list[SentenceChain]
        The original set of chains

    clustering: list[int]
        A list of cluster labels. One label for each chain in ```chains```. This is the result of ```chain_clustering```
    
    save_to: str, optional
        Path to save the plot to. By default, the path is ```None``` and the plot does not get saved

    show: bool
        Whether to display the plot on the screen or not. Defaults to ```False```

    ax: Axes
        We can provide an optional subplot to plot on instead of generating the figure inside the function.
        If this argument is provided, then ```save_to``` and ```show``` are ignored

    return_legend: bool
        If set to ```True```, then the legend elements are returned from the function instead of
        drawn on the axis object. Defaults to ```False```
    '''

    #Parameters checks
    #-----------------------------------------------------------------------
    if ax is not None and show:
        warnings.warn("A custom 'ax' object was provided, therefore setting 'show' to True will be ignored")
        show = False
    if ax is not None and save_to is not None:
        warnings.warn("A custom 'ax' object was provided, therefore setting 'save_to' to True will be ignored")
        save_to = None

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    #Creating the colors
    #-----------------------------------------------------------------------
    cmap = plt.cm.get_cmap("tab20").colors
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) #when using seed
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        colors = [cmap[(2*label + int(label > 9))%20] if label >= 0 else (0,0,0) for label in clustering_labels]

        #Dimensionality reduction
        #-----------------------------------------------------------------------
        matrix = np.array([chain.vector for chain in chains])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            visualization_reducer = UMAP(n_components=2, metric="cosine", random_state=42, min_dist=min_dista)
        
        reduced = visualization_reducer.fit_transform(matrix)

        #Legend creation
        #-----------------------------------------------------------------------
        n_clusters = len(list(set(clustering_labels)))
        legend_elements = [Patch(facecolor=(0,0,0), label="Outliers")]
        legend_elements += [Patch(facecolor=cmap[(2*i + int(i > 9))%20], label=f'Cluster {i:02}') for i in range(n_clusters-1)]

        #Legend creation
        #-----------------------------------------------------------------------
        ax.scatter(reduced[:, 0], reduced[:, 1], c=colors)
        if not return_legend:
            ax.legend(handles=legend_elements)
        ax.set_xticks([])
        ax.set_yticks([])

        #End
        #-----------------------------------------------------------------------
        if save_to:
            fig.savefig(save_to)

        if show:
            plt.show()
        
        if fig:
            plt.clf()
            plt.close(fig)

        if return_legend:
            return legend_elements

#===================================================================================================

def cluster_mask(clusters: dict[int, ChainCluster]) -> list[int]:
    '''
    Assigns each of the document's initial sentences to one of the clusters

    Arguments
    ---
    clusters: list[ChainCluster]
        The clustering returned by ```chain_clustering```.
        It is important that the unaltered output of the clustering function is passed here, as it assumes a specific ordering

    Returns
    ---
    mask: list[int]
        The sentence clustering
    '''

    #Shows the current chain we're examining within the cluster
    #Remember that in each cluster chains are in order of appearance in the document
    positions = [0 for _ in range(len(clusters))]

    #The next sentence we're looking for
    current_offset = 0

    #A list holding the label for each sentence in the document, in order of appearance
    result = []

    reached_end = False

    while not reached_end:
        #We will know we reached the end when the next sentence offset cannot be found
        #(because it is beyond the document's limits, indicating we reached the end)
        reached_end = True

        for label, cluster in clusters.items():

            try:
                cur_chain = cluster[positions[label]]
            except IndexError:
                continue

            #If this cluster's current chain contains the next sentence we're looking for
            if cur_chain.offset == current_offset:
                reached_end = False
                
                #For each sentence in that chain, we must append this cluster's label to the result
                for _ in range(len(cur_chain)):
                    result.append(label)

                current_offset += len(cur_chain)
                positions[label] += 1

                #Continue the search from the beginning
                break

    return result