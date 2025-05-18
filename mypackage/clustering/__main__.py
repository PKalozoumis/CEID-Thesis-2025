from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from ..sentence import doc_to_sentences, iterative_merge
from .clustering import chain_clustering
from ..elastic import Session, elasticsearch_client, ScrollingCorpus, ElasticDocument, Document
import numpy as np
from sentence_transformers import SentenceTransformer
import json

if __name__ == "__main__":
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    doc = Document.from_json("cached_docs/doc_0000.json", text_path="article")

    sentences = doc_to_sentences(doc, model)
    merged = iterative_merge(sentences, threshold=0.6, round_limit=None, pooling_method="average")
    print(chain_clustering(merged))
