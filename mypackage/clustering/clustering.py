from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from mypackage.sentence import SentenceChain
import numpy as np

def chain_clustering(chains: list[SentenceChain]):
    model = HDBSCAN(min_cluster_size=3, min_samples=5,metric="cosine", store_centers="medoid")
    matrix = np.array([chain.vector for chain in chains])
    clustering = model.fit(matrix)
    print(clustering.labels_)

