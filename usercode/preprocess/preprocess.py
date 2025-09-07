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
    parser.add_argument("-d", action="store", type=str, default=None, help="Comma-separated list of docs or document range. Leave blank for a predefined set of test documents. -1 for all")
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Comma-separated list of index names")
    #parser.add_argument("-m", "--model", action="store", type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Name or alias of the sentence embedding model to use")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    group.add_argument("-t", "--temp", nargs="?", action="store", type=str, default=None, help="Name of a temporary experiment that will be generated as a copy of another experiment") #useful when wanting to test temp changes in code, instead of just parameters
    
    parser.add_argument("--from", action="store", type=str, dest="temp_source", default="default", help="Used with -t. Experiment from which to create the temporary experiment")
    parser.add_argument("-c", action="store", type=str, default=None, help="An optional comment appended the created pickle files")
    parser.add_argument("--cache", action="store_true", default=False, help="Retrieve docs from cache instead of elasticsearch")
    parser.add_argument("-nprocs", action="store", type=int, default=1, help="Number of processes")
    parser.add_argument("-dev", "--device", action="store", type=str, default="cpu", choices=["cpu", "gpu"], help="Device for the embedding model")
    parser.add_argument("-db", action="store", type=str, default='mongo', help="Database to store the preprocessing results in", choices=['mongo', 'pickle'])
    parser.add_argument("--spawn", action="store_true", default=False, help="Set process start method to 'spawn'")
    parser.add_argument("-l", "--limit", action="store", type=int, default=None, help="Document limit for scrolling corpus")
    parser.add_argument("-b", "--batch-size", action="store", type=int, default=2000, help="Batch size for scrolling corpus")
    parser.add_argument("-st", "--scroll-time", action="store", type=str, default="1000s", help="Scroll time for scrolling corpus")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-w", "--warnings", action="store_true", default=False, help="Show warnings")
    parser.add_argument("--no-times", action="store_true", default=False, help="Do not store times")

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--cache-embeddings", action="store_true", default=False, help="Cache sentence embeddings")
    group2.add_argument("--use-embedding-cache", action="store_true", default=False, help="Use cached sentence embeddings")

    parser.add_argument("-a", "--append", action="store_true", default=False, help="Insert document into database without deleting previous results")
    
    args = parser.parse_args()

#==============================================================================================

from rich.console import Console
from functools import partial

import traceback
import warnings
from sentence_transformers import SentenceTransformer

from mypackage.elastic import ElasticDocument, Session, ScrollingCorpus
from mypackage.sentence import doc_to_sentences, chaining, sentence_transformer_from_alias
from mypackage.clustering import chain_clustering
from mypackage.helper import DEVICE_EXCEPTION, batched
import pickle
from collections import namedtuple
from multiprocessing import Process, Pool
import torch.multiprocessing as mp
import json

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes

from mypackage.experiments import ExperimentManager
from rich.rule import Rule
from rich.progress import track, Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from mypackage.storage import PickleSession, DatabaseSession, MongoSession
import re
import time
import pandas as pd

if not args.warnings:
    warnings.filterwarnings("ignore")

console = Console()

db = None

#==============================================================================================

def initializer(p,i,_args,cpu_model = None):
    global params, index_name, model, args, db
    params=p
    index_name=i
    args = _args

    if cpu_model is None:
        #Initialize the model in each one of the pool's processes
        #This is the only way to use model in GPU, as sharing is not allowed
        model = SentenceTransformer(params['sentence_model'], device='cuda')
    else:
        model = cpu_model

    if db is None:
        if args.db == "pickle":
            db = PickleSession(os.path.join("..", "experiments", index_name, "pickles"))
            db.sub_path = params['name']

        elif args.db == "mongo":
            db = MongoSession(db_name=f"experiments_{index_name}")
            db.sub_path = params['name']
            if not args.append:
                db.delete() #Drop the collection if it exists

    #Mark experiment as temporary by creating a hidden file in the experiment directory
    if args.temp is not None: 
        db.set_temp()
        db.is_temp()

#==============================================================================================

#The preprocessing steps
def work(doc: ElasticDocument):
    global params, index_name, model, db, args

    record = {
        'doc': doc.id,
        'exp': params['name'],
        'sent_t': None,
        'chain_t': None,
        'umap_t': None,
        'cluster_t': None
    }

    cache_path = os.path.join("embedding_cache", index_name, params['sentence_model'].replace("/", "-"))

    #Try to load embeddings from cache first
    if args.use_embedding_cache:
        try:
            with open(os.path.join(cache_path, f"{doc.id}.pkl"), "rb") as f:
                sentence_vectors = pickle.load(f)
            sentences = doc_to_sentences(doc, model, sep="\n", precalculated_embeddings=sentence_vectors)
            record['sent_t'] = 0
        except:
            return None

    #If cache is disabled, calculate on the spot
    else:
        record['sent_t'] = time.time()
        if args.verbose: console.print(f"Document {doc.id:02}: Creating sentences...")
        sentences = doc_to_sentences(doc, model, sep="\n")
        record['sent_t'] = round(time.time() - record['sent_t'], 3)
        if args.verbose: console.print(f"Document {doc.id:02}: Created {len(sentences)} sentences")

    #Save to embedding cache
    if args.cache_embeddings:
        os.makedirs(cache_path, exist_ok=True)
        with open(os.path.join(cache_path, f"{doc.id}.pkl"), "wb") as f:
            pickle.dump([s.vector for s in sentences], f)
    
    #Chaining parameters
    record['chain_t'] = time.time()
    if args.verbose: console.print(f"Document {doc.id:02}: Creating chains...")
    merged = chaining(params['chaining_method'])(
        sentences,
        threshold=params['threshold'],
        round_limit=params['round_limit'],
        pooling_method=params['chain_pooling_method'],
        normalize=params['normalize'],
        min_chains=params['min_chains']
    )

    #Assign index to each chain
    for i,c in enumerate(merged):
        c.index = i

    if args.verbose: console.print(f"Document {doc.id:02}: Created {len(merged)} chains")
    record['chain_t'] = round(time.time() - record['chain_t'], 3)

    #Clustering parameters
    if args.verbose: console.print(f"Document {doc.id:02}: Clustering chains...")
    clustering = chain_clustering(
        merged,
        n_components=min(params['n_components'], len(merged)-2) if not params['allow_hierarchical'] else params['n_components'],
        min_dista=params['min_dista'],
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
        n_neighbors=params['n_neighbors'],
        normalize=params['cluster_normalize'],
        pooling_method=params['cluster_pooling_method'],
        cluster_selection_method=params['cluster_selection_method'],
        allow_hierarchical=params['allow_hierarchical']
    )
    record['umap_t'] = clustering.times['umap_time']
    record['cluster_t'] = clustering.times['cluster_time']
    
    #Save to database
    db.store(clustering, params=params)

    return record

#=============================================================================================================

def serial_execution(progress, experiment, index, args, cpu_model, docs, batch_size):
    console.print("Serial execution")
    initializer(experiment, index, args, cpu_model)
    time_records = []

    for i, batch in enumerate(batched(docs, batch_size)):
        task = progress.add_task(f"Processing documents for batch {i}...", total=batch_size, start=True)

        for doc in batch:
            res = work(doc)
            if res is not None:
                time_records.append(res)
            progress.update(task, advance=1)

    return time_records

#=============================================================================================================

def parallel_execution(progress, experiment, index, args, cpu_model, docs, batch_size):
    console.print(f"Starting multiprocessing pool with {args.nprocs} processes\n")
    time_records = []
    if args.spawn:
        mp.set_start_method('spawn', force=True)
    try:
        with Pool(processes=args.nprocs, initializer=initializer, initargs=(experiment, index, args, cpu_model)) as pool:
            
            #Receive a batch of documents from Elasticsearch
            #Pass the workload to the pool
            #The workload should be higher than the pool size
            for i, batch in enumerate(batched(docs, args.batch_size)):
                workload = []

                #Retrieve documents
                #Remove the unserializeable session object from them
                for doc in batch:
                    try:
                        doc.get()
                        doc.session = None
                        workload.append(doc)
                    except Exception:
                        continue

                task = progress.add_task(f"Processing documents for batch {i}...", total=len(workload), start=True)

                #Distribute work
                for res in pool.imap_unordered(work, workload):
                    if res is not None:
                        time_records.append(res)
                    progress.update(task, advance=1)

            return time_records
        
    except BaseException as e:
        pool.terminate()
        pool.join()
        raise e
    
#=============================================================================================================

if __name__ == "__main__":

    exp_manager = ExperimentManager("../common/experiments.json")

    indexes = args.i.split(",")
    if len(indexes) > 1:
        if args.d is not None:
            raise DEVICE_EXCEPTION("THE DOCUMENTS MUST CHOOSE... TO EXIST IN ALL, IT INVITES FRACTURE.")

    if args.temp is not None:
        existing_experiments = exp_manager.experiment_names()

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

        sess = Session(index, base_path="../common", cache_dir="../cache", use= ("cache" if args.cache else "client"))
        docs = exp_manager.get_docs(args.d, sess, scroll_batch_size=args.batch_size, scroll_time=args.scroll_time, scroll_limit=args.limit)

        console.print("Session info:")
        #console.print({'index_name': index, 'docs': docs_to_retrieve})
        console.print(f"DEVICE: {args.device}")
        print()

        #Iterate over the requested experiments
        #For each experiment, we execute it on the requested documents and store the pickle files
        for THIS_NEXT_EXPERIMENT in exp_manager.select_experiments(args.x.split(','), iterable = True):
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

            #Check existence and resolve the model's true name
            THIS_NEXT_EXPERIMENT['sentence_model'] = sentence_transformer_from_alias(THIS_NEXT_EXPERIMENT['sentence_model'], "../common/model_aliases.json")

            cpu_model = None
            if args.device == "cpu":
                cpu_model = SentenceTransformer(THIS_NEXT_EXPERIMENT['sentence_model'], device='cpu')

            #procs = []

            t = time.time()

            #If any exception occurs, we terminate execution for the current index, while keeping the times we've stored so far
            #After the exception, times still get printed
            try:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    disable=args.verbose
                ) as progress:
                    
                    batch_size = args.batch_size if args.d == "-1" else len(docs)

                    #Execution
                    if args.nprocs == 1:
                        time_records = serial_execution(progress, THIS_NEXT_EXPERIMENT, index, args, cpu_model, docs, batch_size)
                    else:
                        time_records = parallel_execution(progress, THIS_NEXT_EXPERIMENT, index, args, cpu_model, docs, batch_size)
                    
                    #Store results
                    if len(time_records) > 0:
                        df = pd.DataFrame(time_records)
                        df.sort_values(by="doc", inplace=True, ascending=True)
                        df.set_index(['doc', 'exp'], inplace=True)
                        print(df)
                        print()
                        os.makedirs("times", exist_ok=True)
                        if not args.no_times:
                            df.to_csv(f"times/preprocessing_results_{time.time()}.csv", index=True)

            except Exception as e:
                traceback.print_exc()
            except KeyboardInterrupt:
                pass
            finally:
                console.print(f"\nTotal time: [cyan]{round(time.time() - t, 3):.3f}s[/cyan]\n")
                continue