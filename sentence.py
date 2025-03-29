from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from kmedoids import KMedoids
from kneed import KneeLocator
from matplotlib import pyplot as plt
from elastic import ScrollingCorpus, elasticsearch_client
from itertools import groupby
from elastic import Document

class Sentence():
    def __init__(self, doc: str | Document, transformer: SentenceTransformer | None = None):
        '''
        **doc**: The
        '''

#============================================================================================

def cosine_sim(vec1, vec2) -> float:
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

#============================================================================================

def doc_to_sentences(sentence_transformer: SentenceTransformer, doc: str):
    sentences = doc.split(".")
    if sentences[-1] == '':
        #print("Removed empty sentence")
        sentences = sentences[:-1]

    embeddings = sentence_transformer.encode(sentences)
    return embeddings, sentences

#============================================================================================

def sentence_clustering(embeddings):
    '''
    Returns:
    - Labels
    - Medoids
    '''
    dista = cosine_distances(embeddings)

    inertia = []
    K_range = list(range(1, len(embeddings)))

    #Find optimal cluster count
    for k in K_range:
        clustering = KMedoids(n_clusters=k, metric="precomputed")
        clustering_model = clustering.fit(dista)
        inertia.append(clustering_model.inertia_)
        print(clustering_model.inertia_)

    knee_locator = KneeLocator(K_range, inertia, curve="convex", direction="decreasing")
    optimal_k = knee_locator.elbow
    optimal_k = 3
    print(optimal_k)

    fig, ax = plt.subplots()
    ax.plot(K_range, inertia, "ro--")
    #plt.show()

    #Cluster optimal
    clustering = KMedoids(n_clusters=int(optimal_k), metric="precomputed")
    clustering_model = clustering.fit(dista)
    medoids = clustering_model.medoid_indices_
    print(f"Clustering: {clustering_model.labels_}")
    print(f"Medoids: {medoids}")

    return clustering_model.labels_, medoids

    #sorted_sentences = sorted(sentences, key=lambda x: query_sim[sentences.index(x)], reverse=True)
    #print(sorted_sentences)

#============================================================================================

if __name__ == "__main__":
    client = elasticsearch_client()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    corpus = ScrollingCorpus(client, "arxiv-index", batch_size=10, doc_field="article")

    for doc in corpus:
        embeddings, sentences = doc_to_sentence_embeddings(model, doc)
        labels, medoids = sentence_clustering(embeddings)
        sorted_data = sorted(zip(labels, sentences), key=lambda x: x[0])
        clusters = {k: [v for _, v in g] for k, g in groupby(sorted_data, key=lambda x: x[0])}
        print(clusters)
        break