from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from mypackage.sentence import SentenceChain
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from umap import UMAP

matplotlib.use("TkAgg")

def chain_clustering(chains: list[SentenceChain]):

    matrix = np.array([chain.vector for chain in chains])

    #Reduce dimensionality before clustering
    clustering_reducer = UMAP(n_components=5, metric="cosine", random_state=42)
    reduced_matrix = clustering_reducer.fit_transform(matrix)

    #Cluster
    model = HDBSCAN(min_cluster_size=3, min_samples=5,metric="cosine", store_centers="medoid")
    clustering = model.fit(reduced_matrix)
    print(clustering.labels_)
    
    #Visualize
    colors = [sns.color_palette()[label] if label >= 0 else (0,0,0) for label in clustering.labels_]
    visualization_reducer = UMAP(n_components=2, metric="cosine")
    reduced = visualization_reducer.fit_transform(matrix)

    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors)
    plt.show()

