from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from mypackage.sentence import SentenceChain, doc_to_sentences, iterative_merge
from mypackage.clustering import chain_clustering
from mypackage.elastic import Session, elasticsearch_client, ScrollingCorpus, ElasticDocument
import numpy as np
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":

    session = Session(elasticsearch_client(), "arxiv-index")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    doc = ElasticDocument(session, 0, text_path="article")

    sentences = doc_to_sentences(doc, model)
    merged = iterative_merge(sentences, threshold=0.6, round_limit=None, pooling_method="average")
    chain_clustering(merged)
