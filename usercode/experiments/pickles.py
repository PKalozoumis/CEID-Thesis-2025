#How does chaining affect the clustering result

import sys
import os
sys.path.append(os.path.abspath("../.."))

from rich.console import Console
from functools import partial

from sentence_transformers import SentenceTransformer

from mypackage.elastic import ElasticDocument, Session
from mypackage.sentence import doc_to_sentences, iterative_merge, buggy_merge, chaining
from mypackage.clustering import chain_clustering
import pickle
from collections import namedtuple
from multiprocessing import Process
import argparse
import json

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes

from helper import experiment_wrapper

from mypackage.storage import save_clusters

console = Console()

#==============================================================================================

def work(doc: ElasticDocument, model: SentenceTransformer, params: dict, index_name: str):
    console.print(f"Document {doc.id:02}: Creating sentences...")
    sentences = doc_to_sentences(doc, model)
    console.print(f"Document {doc.id:02}: Creating chains...")
    merged = chaining(params['chaining_method'])(sentences, threshold=params['threshold'], round_limit=params['round_limit'], pooling_method=params['pooling_method'])
    console.print(f"Document {doc.id:02}: Created {len(merged)} chains")
    console.print(f"Document {doc.id:02}: Clustering chains...")
    labels, clusters = chain_clustering(merged, n_components=params['n_components'], min_dista=params['min_dista'])
    
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
    parser.add_argument("-docs", action="store", type=str, default=None, help="Comma-separated list of docs")
    parser.add_argument("-i", action="store", type=str, default="pubmed-index", help="Index name", choices=[
        "pubmed-index",
        "arxiv-index"
    ])
    parser.add_argument("name", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    args = parser.parse_args()

    #-------------------------------------------------------------------------------------------

    if not args.docs:
        if args.i == "pubmed-index":
            docs_to_retrieve = [1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 6415]
        elif args.i == "arxiv-index":
            docs_to_retrieve = list(range(10))
    else:
        docs_to_retrieve = [int(x) for x in args.docs.split(",")]

    #-------------------------------------------------------------------------------------------

    console.print("Session info:")
    console.print({'index_name': args.i, 'docs': docs_to_retrieve})
    print()

    for THIS_NEXT_EXPERIMENT in experiment_wrapper(args.name.split(','), strict_iterable=True):
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
