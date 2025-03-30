import sys
import os
sys.path.append(os.path.abspath(".."))

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from kmedoids import KMedoids
from kneed import KneeLocator
from matplotlib import pyplot as plt
from elastic import ScrollingCorpus, elasticsearch_client, ElasticDocument, Session, Document
from itertools import pairwise, starmap, chain
from dataclasses import dataclass
from numpy import ndarray
from helper import panel_print
import time
from abc import ABC, abstractmethod
from operator import attrgetter
import json
from functools import cached_property
from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown

#============================================================================================

class SentenceLike(ABC):
    @property
    @abstractmethod
    def vector(self):
        pass

    @property
    @abstractmethod
    def text(self):
        pass

    def similarity(self, other) -> float:
        return np.dot(self.vector, other.vector)/(np.linalg.norm(self.vector)*np.linalg.norm(other.vector))

#============================================================================================    
    
@dataclass(repr=False)
class SimilarityPair:
    '''
    Pair of CosineComparable objects, along with their similarity score
    '''
    s1: SentenceLike
    s2: SentenceLike
    sim: float

    @classmethod
    def from_sentences(cls, s1: SentenceLike, s2: SentenceLike):
        return cls(s1, s2, s1.similarity(s2))

    def __post_init__(self):
        if not isinstance(self.s1, SentenceLike):
            raise ValueError("s1 must be a Sentence or SentenceChain")
        
        if not isinstance(self.s2, SentenceLike):
            raise ValueError("s2 must be a Sentence or SentenceChain")
    
#============================================================================================

@dataclass(repr=False)
class Sentence(SentenceLike):
    _text: str
    _vector: ndarray
    doc: ElasticDocument

    def __str__(self):
        return self.text
    
    @property
    def vector(self):
        return self._vector
    
    @property
    def text(self):
        return self._text
    
#============================================================================================
    
class SentenceChain(SentenceLike):
    '''
    Represents a chain of one or more sequential sentences that are very similar
    '''
    @staticmethod
    def pooling_average(sentences: list[Sentence]):
        return np.average(np.row_stack([s.vector for s in sentences]), axis=0)
    
    @staticmethod
    def pooling_max(sentences: list[Sentence]):
        return np.max(np.row_stack([s.vector for s in sentences]), axis=0)

    @staticmethod
    def pooling(sentences: list[Sentence], pooling_method: str):
        match pooling_method:
            case "average": return SentenceChain.pooling_average(sentences)
            case "max": return SentenceChain.pooling_max(sentences)

    def __init__(self, sentences: list[Sentence], pooling_method: str = "average"):
        self.sentences = sentences
        self._vector = SentenceChain.pooling(self.sentences, pooling_method)

    def __iter__(self):
        return iter(self.sentences)
    
    def __len__(self):
        return len(self.sentences)
    
    def __str__(self):
        return json.dumps([s.text for s in self.sentences])
    
    @property
    def vector(self):
        return self._vector
    
    @cached_property
    def text(self):
        return " ".join([s.text for s in self.sentences])

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
        if pair.sim >= threshold: #Add to the chain
            if i == 0:
                chains.append([pair.s1, pair.s2])
            else:
                #We have already examined s1
                chains[-1] += [pair.s2]

        else: #Create new chain for this sentence
            if i == 0:
                chains.append([pair.s1, pair.s2])
            else:
                #We have already examined s1
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
        print_pairs(merged)

        '''
        labels, medoids = sentence_clustering(embeddings)
        sorted_data = sorted(zip(labels, sentences), key=lambda x: x[0])
        clusters = {k: [v for _, v in g] for k, g in groupby(sorted_data, key=lambda x: x[0])}
        print(clusters)'
        '''
        break