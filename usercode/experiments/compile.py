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
from mypackage.helper import DEVICE_EXCEPTION
import pickle
from collections import namedtuple
from multiprocessing import Process
import argparse
import json

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes

from helper import experiment_wrapper, CHOSEN_DOCS, all_experiments
from rich.rule import Rule
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
    clustering = chain_clustering(
        merged,
        n_components=params['n_components'],
        min_dista=params['min_dista'],
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
        n_neighbors=params['n_neighbors'],
        normalize=params['cluster_normalize'],
        pooling_method=params['cluster_pooling_method']
    )
    
    path = os.path.join(index_name, "pickles", params['name'])
    '''
    with open(os.path.join(index_name, "pickles", params['name'], f"{doc.id}.pkl"), "wb") as f:
        doc.session.client = None #Prepare for pickling
        pickle.dump(ProcessedDocument(doc, merged, labels, clusters), f)
    '''
    save_clusters(clustering, path, params=params)

#=============================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the preprocessing step for the specified documents, and with the specified parameters (experiments)")
    parser.add_argument("-d", action="store", type=str, default=None, help="Comma-separated list of docs")
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Comma-separated list of index names")
    parser.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("--temp", action="store_true", default=False, help="Recompile the default experiment into a new temporary experiment named by -x") #useful when wanting to test temp changes in code, instead of just parameters
    parser.add_argument("-c", action="store", type=str, default=None, help="An optional comment appended the created pickle files")
    parser.add_argument("--no-cache", action="store_true", default=False, help="Disable cache. Retrieve docs directly from Elasticsearch")
    args = parser.parse_args()

    indexes = args.i.split(",")

    if len(indexes) > 1:
        if args.d is not None:
            raise DEVICE_EXCEPTION("THE DOCUMENTS MUST CHOOSE... TO EXIST IN ALL, IT INVITES FRACTURE.")

    if args.temp:
        if args.x == "default":
            raise DEVICE_EXCEPTION("PLEASE GIVE IT A NAME")
        elif args.x in set(all_experiments(names_only=True)):
            raise DEVICE_EXCEPTION("DOES IT NOT SEEK AN IDENTITY OF ITS OWN?")
        elif len(args.x.split(",")) > 1:
            raise DEVICE_EXCEPTION("SPLIT IN NAME, AND IN PURPOSE")
        
        temp_name = args.x
        args.x = "default"

    #-------------------------------------------------------------------------------------------
    for index in indexes:
        console.print(f"\nRunning for index '{index}'")
        console.print(Rule())

        if not args.d:
            docs_to_retrieve = CHOSEN_DOCS.get(index, list(range(10)))
        else:
            docs_to_retrieve = [int(x) for x in args.d.split(",")]

        #-------------------------------------------------------------------------------------------

        console.print("Session info:")
        console.print({'index_name': index, 'docs': docs_to_retrieve})
        print()

        #Iterate over the requested experiments
        #For each experiment, we execute it on the requested documents and store the pickle files
        for THIS_NEXT_EXPERIMENT in experiment_wrapper(args.x.split(','), strict_iterable=True, must_exist=True):
            if args.temp:
                THIS_NEXT_EXPERIMENT['name'] = temp_name
                THIS_NEXT_EXPERIMENT['title'] = temp_name
            THIS_NEXT_EXPERIMENT['temp'] = args.temp

            if args.c:
                THIS_NEXT_EXPERIMENT['comment'] = args.c
            
            console.print(f"Running experiment '{THIS_NEXT_EXPERIMENT['name']}' in '{index}'")
            console.print(THIS_NEXT_EXPERIMENT)

            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
            sess = Session(index, base_path="..", cache_dir="../cache", use= ("client" if args.no_cache else "cache"))
            docs = list(map(partial(ElasticDocument, sess, text_path="article"), docs_to_retrieve))

            #We need to process these documents in parallel
            #We need to create the chains, as well as cluster them
            dir_path = os.path.join(sess.index_name, "pickles", THIS_NEXT_EXPERIMENT['name'])
            os.makedirs(dir_path, exist_ok=True)
            if args.temp: #Mark experiment as temporary
                open(os.path.join(dir_path, ".temp"), "w").close()

            procs = []
            
            for i, doc in enumerate(docs):
                p = Process(target=work, args=(doc,model,THIS_NEXT_EXPERIMENT, index))
                p.start()
                procs.append(p)

            for p in procs:
                p.join()
