from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from mypackage.sentence import SentenceChain
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from umap import UMAP

def chain_clustering(chains: list[SentenceChain]):
    model = HDBSCAN(min_cluster_size=3, min_samples=5,metric="cosine", max_cluster_size=50, store_centers="medoid")
    matrix = np.array([chain.vector for chain in chains])
    clustering = model.fit(matrix)
    print(clustering.labels_)
    
    colors = [sns.color_palette()[label] if label >= 0 else (0,0,0) for label in clustering.labels_]

    reducer = UMAP(n_components=2, metric="cosine")
    reduced = reducer.fit_transform(matrix)

    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors)
    plt.show()

