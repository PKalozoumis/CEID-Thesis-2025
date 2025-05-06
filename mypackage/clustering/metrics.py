
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from ..sentence import SentenceChain
import numpy as np

def chain_clustering_silhouette_score(chains: list[SentenceChain], labels: list[int]):

    #Filter out outliers
    chains = [chain for chain, label in zip(chains, labels) if label >= 0]
    labels = [label for label in labels if label >= 0]

    #From each chain in the list, get its representative vector
    #Crete a matrix from these vectors
    mat = np.array([chain.vector for chain in chains])
    sim = cosine_similarity(mat)

    score = silhouette_score(chains, labels)
    print(score)