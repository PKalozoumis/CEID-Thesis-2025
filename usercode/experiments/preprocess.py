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
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    group.add_argument("-t", "--temp", nargs="?", action="store", type=str, default=None, help="Name of a temporary experiment that will be generated as a copy of another experiment") #useful when wanting to test temp changes in code, instead of just parameters
    
    parser.add_argument("--from", action="store", type=str, dest="temp_source", default="default", help="Used with -t. Experiment from which to create the temporary experiment")
    parser.add_argument("-c", action="store", type=str, default=None, help="An optional comment appended the created pickle files")
    parser.add_argument("--no-cache", action="store_true", default=False, help="Disable cache. Retrieve docs directly from Elasticsearch")
    parser.add_argument("-nprocs", action="store", type=int, default=1, help="Number of processes")
    parser.add_argument("-dev", "--device", action="store", type=str, default="cpu", choices=["cpu", "gpu"], help="Device for the embedding model")
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
from multiprocessing import Process, Pool
import torch.multiprocessing as mp
import json

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes

from helper import experiment_wrapper, CHOSEN_DOCS, all_experiments
from rich.rule import Rule
from mypackage.storage import PickleSession

import time
import pandas as pd

console = Console()

db = None

#==============================================================================================

def initializer(p,i, cpu_model = None):
    global params, index_name, model, db
    params=p
    index_name=i

    if cpu_model is None:
        #Initialize the model in each one of the pool's processes
        #This is the only way to use model in GPU, as sharing is not allowed
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    else:
        model = cpu_model

    if db is None:
        db = PickleSession()
    db.base_path = os.path.join(index_name, "pickles")
    db.sub_path = params['name']

#==============================================================================================

#The preprocessing steps
def work(doc: ElasticDocument):
    global params, index_name, model, db

    record = {
        'doc': doc.id,
        'exp': params['name'],
        'sent_t': None,
        'chain_t': None,
        'umap_t': None,
        'hdbscan_t': None
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
    record['umap_t'] = clustering.times['umap_time']
    record['hdbscan_t'] = clustering.times['cluster_time']
    
    #Save to database
    db.store(clustering, params=params)

    return record

#=============================================================================================================

if __name__ == "__main__":

    indexes = args.i.split(",")
    if len(indexes) > 1:
        if args.d is not None:
            raise DEVICE_EXCEPTION("THE DOCUMENTS MUST CHOOSE... TO EXIST IN ALL, IT INVITES FRACTURE.")

    if args.temp is not None:
        existing_experiments = set(all_experiments(names_only=True))

        if args.temp in existing_experiments:
            raise DEVICE_EXCEPTION("DOES IT NOT SEEK AN IDENTITY OF ITS OWN?")
        if args.temp_source not in existing_experiments:
             raise DEVICE_EXCEPTION("IT IS BARREN AND CANNOT BE COPIED")

        #For temporary experiments, the experiment to be loaded becomes temp_source
        args.x = args.temp_source

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
        console.print(f"DEVICE: {args.device}")
        print()

        time_records = []

        #Iterate over the requested experiments
        #For each experiment, we execute it on the requested documents and store the pickle files
        for THIS_NEXT_EXPERIMENT in experiment_wrapper(args.x.split(','), strict_iterable=True, must_exist=True):
            if args.temp is not None:
                THIS_NEXT_EXPERIMENT['name'] = args.temp
                THIS_NEXT_EXPERIMENT['title'] = args.temp
                THIS_NEXT_EXPERIMENT['temp'] = True
                THIS_NEXT_EXPERIMENT['temp_source'] = args.temp_source
            else:
                THIS_NEXT_EXPERIMENT['temp'] = False

            if args.c:
                THIS_NEXT_EXPERIMENT['comment'] = args.c
            
            console.print(f"Running experiment '{THIS_NEXT_EXPERIMENT['name']}' in '{index}'")
            console.print(THIS_NEXT_EXPERIMENT)

            #Load sentence transformer and elastic session
            sess = Session(index, base_path="..", cache_dir="../cache", use= ("client" if args.no_cache else "cache"))

            cpu_model = None
            if args.device == "cpu":
                cpu_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

            docs = [ElasticDocument(sess, doc, text_path="article") for doc in docs_to_retrieve]

            #Path where the experiments will be stored
            dir_path = os.path.join(sess.index_name, "pickles", THIS_NEXT_EXPERIMENT['name'])
            os.makedirs(dir_path, exist_ok=True)

            #Mark experiment as temporary by creating a hidden file in the experiment directory
            if args.temp is not None: 
                open(os.path.join(dir_path, ".temp"), "w").close()

            procs = []

            console.print(f"Starting multiprocessing pool with {args.nprocs} processes\n")

            t = time.time()

            '''
            #Serial test
            initializer(THIS_NEXT_EXPERIMENT, index, cpu_model)
            for doc in docs:
                console.print(work(doc))
            '''

            #mp.set_start_method('spawn', force=True)
            with Pool(processes=args.nprocs, initializer=initializer, initargs=(THIS_NEXT_EXPERIMENT, index, cpu_model)) as pool:
                for res in pool.imap_unordered(work, docs):
                    #console.print(res)
                    time_records.append(res)

            console.print(f"\nTotal time: {round(time.time() - t, 3):.3f}s\n")

        df = pd.DataFrame(time_records)
        df.set_index(['doc', 'exp'], inplace=True)
        print(df)
        print()

        
