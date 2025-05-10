import sys
import os
sys.path.append(os.path.abspath(".."))

from rich.console import Console
from functools import partial

from sentence_transformers import SentenceTransformer

from mypackage.elastic import ElasticDocument, Session
from mypackage.sentence import doc_to_sentences, iterative_merge
from mypackage.clustering import chain_clustering, visualize_clustering
import pickle
from collections import namedtuple
from multiprocessing import Process, set_start_method
import argparse

#set_start_method('spawn', force=True)

ProcessedDocument = namedtuple("ProcessedDocument", ["doc", "chains", "labels", "clusters"])
console = Console()

#==============================================================================================

def work(doc: ElasticDocument, model: SentenceTransformer):
    console.print(f"Document {doc.id:02}: Creating sentences...")
    sentences = doc_to_sentences(doc, model)
    console.print(f"Document {doc.id:02}: Creating chains...")
    merged = iterative_merge(sentences, threshold=0.6, round_limit=None, pooling_method="average")
    console.print(f"Document {doc.id:02}: Created {len(merged)} chains")
    console.print(f"Document {doc.id:02}: Clustering chains...")
    labels, clusters = chain_clustering(merged, n_components=25)
    
    with open(f"pickles/{doc.id}.pkl", "wb") as f:
        pickle.dump(ProcessedDocument(doc, merged, labels, clusters), f)

#=============================================================================================================

def pickles(docs_to_retrieve):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    sess = Session("pubmed-index", cache_dir="cache", use="cache")
    docs = list(map(partial(ElasticDocument, sess, text_path="article"), docs_to_retrieve))

    #We need to process these documents in parallel
    #We need to create the chains, as well as cluster them

    os.makedirs("pickles", exist_ok=True)

    procs = []

    for i, doc in enumerate(docs):
        p = Process(target=work, args=(doc,model))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

#=============================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("-docs", action="store", default=None)
    args = parser.parse_args()

    if not args.docs:
        docs_to_retrieve = [1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 6415]
    else:
        docs_to_retrieve = args.docs.split(",")

    if args.pickle:
        pickles(docs_to_retrieve)
    
    if args.plot:
        pkl = []

        for fname in map(lambda x: f"pickles/{x}.pkl", docs_to_retrieve):
            with open(fname, "rb") as f:
                pkl.append(pickle.load(f))

        os.makedirs("images", exist_ok=True)
        for i, p in enumerate(pkl):
            console.print(f"Plotting document {p.doc.id}")
            visualize_clustering(p.chains, p.labels, save_to=f"images/{i:02}_{p.doc.id}.png", show=False)