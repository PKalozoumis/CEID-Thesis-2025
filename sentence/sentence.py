import sys
import os
sys.path.append(os.path.abspath(".."))

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from kmedoids import KMedoids
from kneed import KneeLocator
from matplotlib import pyplot as plt
from elastic import ScrollingCorpus, elasticsearch_client, Session, Document
from itertools import pairwise, starmap
from helper import panel_print
from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown
from metrics import intra_chain_distance
from classes import *

#============================================================================================

def doc_to_sentences(doc: Document, transformer: SentenceTransformer) -> list[Sentence]:
    '''
    Breaks down a document into sentences. For the entire set of sentences, the embeddings are calculated

    Args:
        doc (Document): The document to extract sentences from
        transformer (SentenceTransformer): The model that will generate the embeddings

    Returns:
        list[Sentence]: A list of Sentence objects
    '''
    sentences = doc.text().split("\n")
    if sentences[-1] == '':
        sentences = sentences[:-1]

    embeddings = transformer.encode(sentences)

    result = []
    for s, e in zip(sentences, embeddings):
        result.append(Sentence(s, e, doc))

    return result

#============================================================================================

def print_pairs(sentences):
    console = Console()
    console.clear()

    table = Table()
    table.add_column("Sentence Pair")
    table.add_column("Similarity", vertical="top")

    for thing in starmap(lambda x, y: (x,y,x.similarity(y)), pairwise(sentences)):
        mytext = f'''
- {thing[0].text.strip()}


- {thing[1].text.strip()}

---
        '''
        table.add_row(Markdown(mytext), "\n\n"+str(thing[2]))

    console.print(table)

#============================================================================================

def iterative_merge(sentences: list[SentenceLike],*, threshold: float, round_limit: int | None = 1, pooling_method="average"):
    pairs = [SimilarityPair.from_sentences(s1, s2) for s1, s2 in pairwise(sentences)]

    #No more merging can happen
    if not any(filter(lambda x: x.sim > threshold, pairs)):
        return sentences

    chains = []

    for i, pair in enumerate(pairs):

        if i == 0:
            chains.append([pair.s1, pair.s2])
            continue
        
        if pair.sim >= threshold: #Add to the chain
            chains[-1].append(pair.s2)
        else: #Create new chain for this sentence
            chains.append([pair.s2])

    result = [SentenceChain(c, pooling_method) for c in chains]
    
    if round_limit is None:
        return iterative_merge(result, threshold=threshold, round_limit=None, pooling_method=pooling_method)
    elif round_limit > 1:
        return iterative_merge(result, threshold=threshold, round_limit=round_limit-1, pooling_method=pooling_method)
    else:
        return result

#============================================================================================

def cosine_sim(vec1, vec2) -> float:
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

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
    session = Session(elasticsearch_client("../credentials.json", "../http_ca.crt"), "arxiv-index")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    corpus = ScrollingCorpus(session, batch_size=10, doc_field="article")

    for doc in corpus:
        sentences = doc_to_sentences(doc, model)
        merged = iterative_merge(sentences, threshold=0.6, round_limit=None, pooling_method="average")
        print(len(merged[0]))
        print(merged[0].sentences)
        panel_print(merged[0].text)
        intra_chain_distance(merged[0])

        '''
        labels, medoids = sentence_clustering(embeddings)
        sorted_data = sorted(zip(labels, sentences), key=lambda x: x[0])
        clusters = {k: [v for _, v in g] for k, g in groupby(sorted_data, key=lambda x: x[0])}
        print(clusters)'
        '''
        break