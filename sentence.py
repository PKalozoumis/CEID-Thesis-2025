from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from kmedoids import KMedoids
from kneed import KneeLocator
from matplotlib import pyplot as plt
from elastic import ScrollingCorpus, elasticsearch_client, Document, Session
from itertools import pairwise, starmap, chain
from dataclasses import dataclass
from numpy import ndarray
from helper import panel_print
import time
from abc import ABC, abstractmethod
from operator import attrgetter
import json
from functools import cached_property

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

    def sim(self, other) -> float:
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
        return cls(s1, s2, s1.sim(s2))

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
    doc: Document

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
    def polling_average(sentences: list[Sentence]):
        return np.average(np.row_stack([s.vector for s in sentences]), axis=0)

    def __init__(self, sentences: list[Sentence], polling_approach: str = "average"):
        self.sentences = sentences
        self._vector = SentenceChain.polling_average(self.sentences)

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
    if doc.text_path is None:
        raise ValueError(f"text_path not specified for Document(id={doc.id})")

    sentences = doc.text().split("\n")
    if sentences[-1] == '':
        sentences = sentences[:-1]

    embeddings = transformer.encode(sentences)

    result = []
    for s, e in zip(sentences, embeddings):
        result.append(Sentence(s, e, doc))

    return result

#============================================================================================

def chain_clustering(sentences: list[Sentence]):
    '''
    Cluster similar neighboring sentneces together

    Args:
        sentences (list[Sentence]): The sentences to cluster

    Returns:
        Clustering
    '''
    sim = starmap(lambda x, y: cosine_sim(x.vector, y.vector), pairwise(sentences))
    print(list(sim))

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
    session = Session(elasticsearch_client(), "arxiv-index")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    corpus = ScrollingCorpus(session, batch_size=10, doc_field="article")

    for doc in corpus:
        sentences = doc_to_sentences(doc, model)
        chain_clustering(sentences)

        '''
        labels, medoids = sentence_clustering(embeddings)
        sorted_data = sorted(zip(labels, sentences), key=lambda x: x[0])
        clusters = {k: [v for _, v in g] for k, g in groupby(sorted_data, key=lambda x: x[0])}
        print(clusters)'
        '''
        break