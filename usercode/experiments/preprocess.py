"""
The preprocessing step of the pipeline
"""

import sys
import os
sys.path.append(os.path.abspath("../.."))

import argparse

#==============================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the preprocessing step for the specified documents, and with the specified parameters (experiments)")
    parser.add_argument("-d", action="store", type=str, default=None, help="Comma-separated list of docs. Leave blank for a predefined set of test documents")
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Comma-separated list of index names")
    parser.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("--temp", action="store_true", default=False, help="Recompile the default experiment into a new temporary experiment named by -x") #useful when wanting to test temp changes in code, instead of just parameters
    parser.add_argument("-c", action="store", type=str, default=None, help="An optional comment appended the created pickle files")
    parser.add_argument("--no-cache", action="store_true", default=False, help="Disable cache. Retrieve docs directly from Elasticsearch")
    parser.add_argument("-nprocs", action="store", type=int, default=1, help="Number of processes")
    args = parser.parse_args()

#==============================================================================================

from rich.console import Console
from functools import partial

from sentence_transformers import SentenceTransformer

from mypackage.elastic import ElasticDocument, Session
from mypackage.sentence import doc_to_sentences, chaining
from mypackage.clustering import chain_clustering
from mypackage.helper import DEVICE_EXCEPTION
import pickle
from collections import namedtuple
#from multiprocessing import Process, Pool
import torch.multiprocessing as mp
import json

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes

from helper import experiment_wrapper, CHOSEN_DOCS, all_experiments
from rich.rule import Rule
from mypackage.storage.store import save_clusters

import time

console = Console()

#==============================================================================================

def initializer(p,i, cpu_model = None):
    global params, index_name, model
    params=p
    index_name=i

    if cpu_model is None:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    else:
        model = cpu_model


#The preprocessing steps
def work(doc: ElasticDocument):
    global params, index_name, model

    record = {
        'doc': doc.id,
        'exp': params['title'],
        'sent_t': None,
        'chain_t': None,
        'cluster_t': None
    }

    record['sent_t'] = time.time()
    console.print(f"Document {doc.id:02}: Creating sentences...")
    sentences = doc_to_sentences(doc, model, remove_duplicates=params['remove_duplicates'])
    record['sent_t'] = round(time.time() - record['sent_t'], 3)
    
    #Chaining parameters
    record['chain_t'] = time.time()
    console.print(f"Document {doc.id:02}: Creating chains...")
    merged = chaining(params['chaining_method'])(
        sentences,
        threshold=params['threshold'],
        round_limit=params['round_limit'],
        pooling_method=params['chain_pooling_method'],
        normalize=params['normalize']
    )
    console.print(f"Document {doc.id:02}: Created {len(merged)} chains")
    record['chain_t'] = round(time.time() - record['chain_t'], 3)

    #Clustering parameters
    record['cluster_t'] = time.time()
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
    record['cluster_t'] = round(time.time() - record['cluster_t'], 3)
    
    #Save to database
    path = os.path.join(index_name, "pickles", params['name'])
    save_clusters(clustering, path, params=params)

    return record

#=============================================================================================================

if __name__ == "__main__":

    indexes = args.i.split(",")
    if len(indexes) > 1:
        if args.d is not None:
            raise DEVICE_EXCEPTION("THE DOCUMENTS MUST CHOOSE... TO EXIST IN ALL, IT INVITES FRACTURE.")

    if args.temp:
        if args.x == "default":
            raise DEVICE_EXCEPTION("PLEASE GIVE THE EXPERIMENT A NAME")
        elif args.x in set(all_experiments(names_only=True)):
            raise DEVICE_EXCEPTION("DOES IT NOT SEEK AN IDENTITY OF ITS OWN?")
        elif len(args.x.split(",")) > 1:
            raise DEVICE_EXCEPTION("SPLIT IN NAME, AND IN PURPOSE")
        
        temp_name = args.x
        args.x = "default"

    #Run for each specified index
    #-------------------------------------------------------------------------------------------
    for index in indexes:
        console.print(f"\nRunning for index '{index}'")
        console.print(Rule())

        #If docs are not specified, then a predefined set of docs is selected
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

            #Load sentence transformer and elastic session
            
            sess = Session(index, base_path="..", cache_dir="../cache", use= ("client" if args.no_cache else "cache"))
            docs = [ElasticDocument(sess, doc, text_path="article") for doc in docs_to_retrieve]

            #Path where the experiments will be stored
            dir_path = os.path.join(sess.index_name, "pickles", THIS_NEXT_EXPERIMENT['name'])
            os.makedirs(dir_path, exist_ok=True)

            #Mark experiment as temporary by creating a hidden file in the experiment directory
            if args.temp: 
                open(os.path.join(dir_path, ".temp"), "w").close()

            procs = []

            console.print(f"Starting multiprocessing pool with {args.nprocs} processes\n")

            t = time.time()

            records = []
            
            mp.set_start_method('spawn', force=True)
            with mp.Pool(processes=args.nprocs, initializer=initializer, initargs=(THIS_NEXT_EXPERIMENT, index)) as pool:
                for res in pool.imap_unordered(work, docs):
                    console.print(res)
                    records.append(res)

            console.print(f"Total time: {round(time.time() - t, 3):.3f}s")
