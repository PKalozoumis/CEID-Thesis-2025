import os
import sys

sys.path.append(os.path.abspath("../.."))

import argparse

#=================================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of preprocessing results")
    parser.add_argument("-d", action="store", type=str, default=None, help="Comma-separated list of docs. Leave blank for a predefined set of test documents. -1 for all")
    parser.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("-i", "--index", action="store", type=str, default="pubmed", help="Index name")
    parser.add_argument("-db", action="store", type=str, default='mongo', help="Database to load the preprocessing results from", choices=['mongo', 'pickle'])
    parser.add_argument("--cache", action="store_true", default=False, help="Retrieve docs from cache instead of elasticsearch")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-store", action="store_true", default=False, help="Do not store metric")
    group.add_argument("--replace", action="store_true", default=False, help="Replace existing file")
    args = parser.parse_args()

#=================================================================================================================

from collections import namedtuple
from mypackage.elastic import Session, ElasticDocument
from mypackage.helper import NpEncoder, create_table, write_to_excel_tab, DEVICE_EXCEPTION, batched
from mypackage.clustering.metrics import chain_clustering_silhouette_score, clustering_metrics
from mypackage.sentence import SentenceChain
from mypackage.clustering import ChainClustering, dimensionality_reducer

import numpy as np
import pandas as pd

from mypackage.experiments import ExperimentManager
from mypackage.storage import PickleSession, MongoSession, DatabaseSession, ProcessedDocument

import pickle
from itertools import chain
from functools import reduce
import json

from rich.console import Console
from rich.rule import Rule
from rich.progress import track, Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from matplotlib import pyplot as plt

from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")


console = Console()

#=================================================================================================================

def initializer(m):
    global reducer, db, metric
    reducer = dimensionality_reducer()
    db = PickleSession() if args.db == "pickle" else MongoSession()
    metric = m

def work(processed):
    global reducer, db, metric

    try:
        if processed is None:
            #console.print(f"Doc {d.id} skipped: No preprocessing data")
            return None
        #Distinct labels
        if len(list(set(processed.clustering.labels))) < 2:
            #console.print(f"Doc {d.id} skipped: No distinct labels")
            return None

        processed.remove_outliers()

        if reducer is not None and len(processed.clustering.chains) < 27:
            if not processed.params['allow_hierarchical']:
                reducer = dimensionality_reducer(n_components=len(processed.clustering.chains)-2)
            elif metric=="dbcv": #Small docs were not clustered with density, and this metric does not apply
                #console.print(f"Doc {d.id} skipped: Cannot apply density-based metric")
                return None
            
        return {'doc': processed.doc.id, **clustering_metrics(processed.clustering, metric, value=True, reducer=reducer)}
    except Exception as e:
        print(f"For document {processed.doc.id}: {e}")
        return None
    
#=================================================================================================================

def gather_processed_batch(batch):
    workload = []
    for processed in db.load(sess, list(batch), skip_missing_docs=True):
        try:
            processed.doc.get()
            processed.doc.session = None
            workload.append(processed)
        except Exception:
            continue

    return workload

#=================================================================================================================

def run_for_metric(metric: str, reducer=None):
    fname = f"{exp}/{metric}.csv"

    if not args.replace and os.path.exists(fname):
        #console.print(f"Found {fname}")
        df = pd.read_csv(fname, index_col=["doc"]).sort_index()[[metric]]
    else:
        scores = []
        
        with Pool(processes=5, initializer=initializer, initargs=(metric,)) as pool:

            batch_size = 200
            all_batches = list(batched(docs, batch_size))
            workload = gather_processed_batch(all_batches[0])
            task = progress.add_task(f"Experiment: [cyan]{exp}[/cyan] Metric: [cyan]{metric}[/cyan]", start=True, total=len(docs))

            for i in range(len(all_batches)):
                #Distribute workload
                results =  pool.imap_unordered(work, workload)

                #While work is being done, gather the next batch
                if i < len(all_batches)-1:
                    workload = gather_processed_batch(all_batches[i+1])

                #Gather results
                for res in results:
                    progress.update(task, advance=1)
                    if res is not None:
                        scores.append(res)

        df = pd.DataFrame(scores).set_index('doc').sort_index()[[metric]]
        if not args.no_store or args.replace:
            df.to_csv(fname)

    console.print(df)
    return df

#=========================================================================================================

if __name__ == "__main__":

    if args.d == -1:
        raise Exception("no")

    exp_manager = ExperimentManager("../common/experiments.json")
    sess = Session(args.index, base_path="../common", cache_dir="../cache", use="cache" if args.cache else "client")
    docs = exp_manager.get_docs(args.d, sess, scroll_batch_size=2000)
    db = PickleSession() if args.db == "pickle" else MongoSession()
    db.base_path = os.path.join(sess.index_name, "pickles") if db.db_type == "pickle" else f"experiments_{sess.index_name}"

    metrics = ['silhouette']
    #metrics = ['dbcv']
    #metrics = ['dbi']

    reducer = dimensionality_reducer()
    #reducer = None

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as progress:
        for exp in db.available_experiments(args.x):
            os.makedirs(f"{exp}", exist_ok=True)
            db.sub_path = exp
            for metric in metrics:
                df = run_for_metric(metric, reducer)
        
    db.close()