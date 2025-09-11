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
    parser.add_argument("-m", "--metrics", action="store", type=str, required=True, help="Comma-separated metrics to calculate")
    parser.add_argument("-b", "--batch-size", action="store", type=int, default=200, help="Number of processed documents per batch")
    parser.add_argument("--cache", action="store_true", default=False, help="Retrieve docs from cache instead of elasticsearch")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-store", action="store_true", default=False, help="Do not store metric")
    group.add_argument("--replace", action="store_true", default=False, help="Replace existing file")
    parser.add_argument("--compare", action="store_true", default=False, help="Enable comparison mode")
    parser.add_argument("--columns", action="store_true", default=False, help="Experiments at the columns")
    args = parser.parse_args()

#=================================================================================================================

from collections import namedtuple
from mypackage.elastic import Session, ElasticDocument
from mypackage.helper import NpEncoder, create_table, write_to_excel_tab, DEVICE_EXCEPTION, batched, format_latex_table, rule_print
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

def cluster_sizes_work(doc: int):
    global db

    pipeline = [
        {"$match": {"id": doc, "label": {"$ne": -1}}},
        {"$project": {"chain_length": {"$size": "$chains"}}}
    ]

    return {
        doc: [cluster["chain_length"] for cluster in db.collection.aggregate(pipeline)]
    }

#-------------------------------------------------------------------

def cluster_sizes_initializer(d: MongoSession):
    global db
    db = d.duplicate_session(d)

#-------------------------------------------------------------------

def cluster_sizes(db, docs: list[int]):

    lengths = {}
    
    with Pool(10, initializer=cluster_sizes_initializer, initargs=(db,)) as pool:
        task = progress.add_task(f"Experiment: [cyan]{exp}[/cyan] cluster sizes", start=True, total=len(docs))
        for res in pool.imap_unordered(cluster_sizes_work, docs):
            lengths |= res
            progress.update(task, advance=1)

    return lengths

#=================================================================================================================

def initializer(m):
    global reducer, db, metrics
    reducer = dimensionality_reducer()
    metrics = m

def work(processed):
    global reducer, metrics

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
            elif "dbcv" in metrics: #Small docs were not clustered with density, and this metric does not apply
                #console.print(f"Doc {d.id} skipped: Cannot apply density-based metric")
                metrics = [m for m in metrics if m != "dbcv"]
            
        return {'doc': processed.doc.id, **clustering_metrics(processed.clustering, metrics, value=True, reducer=reducer)}
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

def run_for_metrics(metrics: list[str], reducer=None):
    metric_dfs = []

    must_calculate = []

    #Check if some of the metrics exist
    for metric in metrics:
        fname = f"metrics/{exp}/{metric}.csv"

        #Check if this metric has already been calculated
        if not args.replace and os.path.exists(fname):
            df = pd.read_csv(fname, index_col=["doc"]).sort_index()[[metric]]
            df = df[df.index.isin([d.id for d in docs])]
            metric_dfs.append(df.agg("median").to_frame(exp).T)

        #Check if this is a special metric
        elif metric == "avg_cluster_size" or metric=="avg_cluster_count": 
                specil = f"metrics/{exp}/cluster_sizes.json"
                if not os.path.exists(specil):
                    lengths = cluster_sizes(db, [d.id for d in docs])
                    with open(specil, "w") as f:
                        json.dump(lengths, f, indent="\t")
                else:
                    with open(specil, "r") as f:
                        lengths = json.load(f)

                if metric == "avg_cluster_size":
                    temp = np.mean(list(chain.from_iterable(l for l in lengths.values())))
                    temp = pd.DataFrame({'exp': exp, metric: temp}, index=["exp"]).set_index("exp")
                    temp.index.names = [None]
                    metric_dfs.append(temp)
                    
                elif metric == "avg_cluster_count":
                    temp = np.mean([len(l) for l in lengths.values()])
                    temp = pd.DataFrame({'exp': exp, metric: temp}, index=["exp"]).set_index("exp")
                    temp.index.names = [None]
                    metric_dfs.append(temp)

        else:
            must_calculate.append(metric)
            

    #------------------------------------------------------------------------------

    if len(must_calculate) > 0:
        calculated_scores = []
        
        with Pool(processes=5, initializer=initializer, initargs=(must_calculate,)) as pool:

            batch_size = args.batch_size
            all_batches = list(batched(docs, batch_size))
            workload = gather_processed_batch(all_batches[0])
            task = progress.add_task(f"Experiment: [cyan]{exp}[/cyan]", start=True, total=len(docs))

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
                        calculated_scores.append(res)

        #Each of the calculated scores must become its own dataframe, and stored in a separate file
        df = pd.DataFrame(calculated_scores).set_index('doc').sort_index()
        for c in df.columns.tolist():
            fname = f"metrics/{exp}/{c}.csv"
            temp_df = df[[c]]
            metric_dfs.append(temp_df.agg("median").to_frame(exp).T)
            if not args.no_store or args.replace:
                temp_df.to_csv(fname)

        progress.stop_task(task)

    return pd.concat(metric_dfs, axis=1)[metrics]

#=========================================================================================================

if __name__ == "__main__":

    if args.d == -1:
        raise Exception("no")

    exp_manager = ExperimentManager("../common/experiments.json")
    sess = Session(args.index, base_path="../common", cache_dir="../cache", use="cache" if args.cache else "client")
    docs = exp_manager.get_docs(args.d, sess, scroll_batch_size=2000)
    db = DatabaseSession.init_db(args.db, exp_manager.db_name(sess.index_name))

    metrics = args.metrics.split(",")

    allowed_metrics = [
        'silhouette',
        'dbcv',
        'flat_silhouette',
        'avg_centroid_sim',
        'avg_cluster_size',
        'avg_cluster_count',
        'avg_flat_centroid_sim',
        'avg_flat_cluster_size'
    ]

    for m in metrics:
        if m not in allowed_metrics:
            raise Exception(f"Unknown metric {m}")

    reducer = dimensionality_reducer()
    #reducer = None

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as progress:
        if args.compare:
            exp_dfs = []
        experiments = db.available_experiments(args.x)
        for exp in experiments:
            os.makedirs(f"metrics/{exp}", exist_ok=True)
            db.sub_path = exp

            metric_dfs = run_for_metrics(metrics, reducer)
            
            
            #For this experiment, merge the metrics into one dataframe
            if args.columns:
                metric_dfs = metric_dfs.T

            #console.print(f"\n{metric_dfs}\n")
            
            if args.compare:
                exp_dfs.append(metric_dfs)

        if args.compare:
            #Each experiment is its own row
            exp_dfs = pd.concat(exp_dfs, axis=0 if not args.columns else 1).round(3)
            console.print(exp_dfs)

            latex = exp_dfs.to_latex(
                escape=True,
                column_format='l' + 'l'*(len(metrics) if not args.columns else len(experiments)),
                caption="Μετρικές συσταδοποίησης", 
                label="tab:cluster", 
                float_format="%.3f",
                position="H"
            )
            format_latex_table(latex, name="Μετρικές συσταδοποίησης")
        
    db.close()