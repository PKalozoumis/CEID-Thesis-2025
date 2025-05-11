#How does chaining affect the clustering result

import sys
import os
sys.path.append(os.path.abspath(".."))

from rich.console import Console
from functools import partial

from sentence_transformers import SentenceTransformer

from mypackage.elastic import ElasticDocument, Session
from mypackage.sentence import doc_to_sentences, iterative_merge, buggy_merge, chaining
from mypackage.clustering import chain_clustering, visualize_clustering
import pickle
from collections import namedtuple
from multiprocessing import Process, set_start_method
import argparse
import json

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes


#set_start_method('spawn', force=True)

ProcessedDocument = namedtuple("ProcessedDocument", ["doc", "chains", "labels", "clusters"])
console = Console()

#==============================================================================================

def work(doc: ElasticDocument, model: SentenceTransformer, params: dict):
    console.print(f"Document {doc.id:02}: Creating sentences...")
    sentences = doc_to_sentences(doc, model)
    console.print(f"Document {doc.id:02}: Creating chains...")
    merged = chaining(params['chaining_method'])(sentences, threshold=params['threshold'], round_limit=params['round_limit'], pooling_method=params['pooling_method'])
    console.print(f"Document {doc.id:02}: Created {len(merged)} chains")
    console.print(f"Document {doc.id:02}: Clustering chains...")
    labels, clusters = chain_clustering(merged, n_components=params['n_components'])
    
    with open(os.path.join("pickles", params['name'], f"{doc.id}.pkl"), "wb") as f:
        pickle.dump(ProcessedDocument(doc, merged, labels, clusters), f)

#=============================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-docs", action="store", type=str, default=None, help="Comma-separated list of docs")
    parser.add_argument("name", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    args = parser.parse_args()

    #-------------------------------------------------------------------------------------------

    if not args.docs:
        docs_to_retrieve = [1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 6415]
    else:
        docs_to_retrieve = args.docs.split(",")

    #-------------------------------------------------------------------------------------------

    for THIS_NEXT_EXPERIMENT in args.name.split(","):
        params = {}
        params['chaining_method'] = 'iterative'
        params['threshold'] = 0.6
        params['round_limit'] = None
        params['pooling_method'] = 'average'
        params['n_components'] = 25
        params['name'] = THIS_NEXT_EXPERIMENT

        if THIS_NEXT_EXPERIMENT != 'default':
            try:
                with open(os.path.join("params", THIS_NEXT_EXPERIMENT + ".json"), "r") as f:
                    params |= json.load(f)
            except:
                print(f"Could not find experiment '{THIS_NEXT_EXPERIMENT}'. Using default params")
                params['name'] = 'default'

        #-------------------------------------------------------------------------------------

        console.print(f"Running experiment '{params['name']}'")
        console.print(params)

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        sess = Session("pubmed-index", cache_dir="cache", use="cache")
        docs = list(map(partial(ElasticDocument, sess, text_path="article"), docs_to_retrieve))

        #We need to process these documents in parallel
        #We need to create the chains, as well as cluster them

        os.makedirs(os.path.join("pickles", params['name']), exist_ok=True)

        procs = []

        for i, doc in enumerate(docs):
            p = Process(target=work, args=(doc,model,params))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
