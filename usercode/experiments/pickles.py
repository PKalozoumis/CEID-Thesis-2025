#How does chaining affect the clustering result

import sys
import os
sys.path.append(os.path.abspath("../.."))

from rich.console import Console
from functools import partial

from sentence_transformers import SentenceTransformer

from mypackage.elastic import ElasticDocument, Session
from mypackage.sentence import doc_to_sentences, chaining
from mypackage.clustering import chain_clustering
import pickle
from collections import namedtuple
from multiprocessing import Process
import argparse
import json

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes

from helper import experiment_wrapper, ARXIV_DOCS, PUBMED_DOCS

from mypackage.storage import save_clusters

console = Console()

#==============================================================================================

def work(doc: ElasticDocument, model: SentenceTransformer, params: dict, index_name: str):
    console.print(f"Document {doc.id:02}: Creating sentences...")
    sentences = doc_to_sentences(doc, model, remove_duplicates=params['remove_duplicates'])
    
    #Chaining parameters
    console.print(f"Document {doc.id:02}: Creating chains...")
    merged = chaining(params['chaining_method'])(
        sentences,
        threshold=params['threshold'],
        round_limit=params['round_limit'],
        pooling_method=params['pooling_method'],
        normalize=params['normalize']
    )
    console.print(f"Document {doc.id:02}: Created {len(merged)} chains")

    console.print(f"Document {doc.id:02}: Clustering chains...")
    labels, clusters = chain_clustering(
        merged,
        n_components=params['n_components'],
        min_dista=params['min_dista'],
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
        n_neighbors=params['n_neighbors']
    )
    
    path = os.path.join(index_name, "pickles", params['name'])
    '''
    with open(os.path.join(index_name, "pickles", params['name'], f"{doc.id}.pkl"), "wb") as f:
        doc.session.client = None #Prepare for pickling
        pickle.dump(ProcessedDocument(doc, merged, labels, clusters), f)
    '''
    save_clusters(clusters, path)

#=============================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", action="store", type=str, default=None, help="Comma-separated list of docs")
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Index name", choices=[
        "pubmed",
        "arxiv"
    ])
    parser.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    args = parser.parse_args()

    args.i += "-index"

    #-------------------------------------------------------------------------------------------

    if not args.d:
        if args.i == "pubmed-index":
            docs_to_retrieve = PUBMED_DOCS
        elif args.i == "arxiv-index":
            docs_to_retrieve = ARXIV_DOCS
    else:
        docs_to_retrieve = [int(x) for x in args.d.split(",")]

    #-------------------------------------------------------------------------------------------

    console.print("Session info:")
    console.print({'index_name': args.i, 'docs': docs_to_retrieve})
    print()

    #Iterate over the requested experiments
    #For each experiment, we execute it on the requested documents and store the pickle files
    for THIS_NEXT_EXPERIMENT in experiment_wrapper(args.x.split(','), strict_iterable=True):
        console.print(f"Running experiment '{THIS_NEXT_EXPERIMENT['name']}'")
        console.print(THIS_NEXT_EXPERIMENT)

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        sess = Session(args.i, base_path="../..", cache_dir="../cache", use="cache")
        docs = list(map(partial(ElasticDocument, sess, text_path="article"), docs_to_retrieve))

        #We need to process these documents in parallel
        #We need to create the chains, as well as cluster them
        os.makedirs(os.path.join(sess.index_name, "pickles", THIS_NEXT_EXPERIMENT['name']), exist_ok=True)
        procs = []
        
        for i, doc in enumerate(docs):
            p = Process(target=work, args=(doc,model,THIS_NEXT_EXPERIMENT, args.i))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
