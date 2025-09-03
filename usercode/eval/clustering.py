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
    parser.add_argument("--store", action="store_true", default=False, help="Store metric")
    args = parser.parse_args()

#=================================================================================================================

from collections import namedtuple
from mypackage.elastic import Session, ElasticDocument
from mypackage.helper import NpEncoder, create_table, write_to_excel_tab, DEVICE_EXCEPTION
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

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")


console = Console()

#=================================================================================================================

def run_for_metric(metric: str, reducer):
    fname = f"{exp}/{metric}.csv"

    if os.path.exists(fname):
        #console.print(f"Found {fname}")
        df = pd.read_csv(fname, index_col=["doc"])
    else:
        scores = []
        for d in docs:
            try:
                processed = db.load(sess, d, skip_missing_docs=True)
                if processed is None:
                    continue
                #Distinct labels
                if len(list(set(processed.clustering.labels))) < 2:
                    continue

                processed.remove_outliers()
                scores.append({'doc': d.id, **clustering_metrics(processed.clustering, metric, value=True, reducer=reducer)})
            except Exception as e:
                print(f"For document {d.id}: {e}")

        df.set_index('doc')
        df = pd.DataFrame(scores)
        if args.store:
            df.to_csv(fname)

    console.print(df)

#=========================================================================================================

if __name__ == "__main__":
    exp_manager = ExperimentManager("../common/experiments.json")
    sess = Session(args.index, base_path="../common", cache_dir="../cache", use="cache" if args.cache else "client")
    docs = exp_manager.get_docs(args.d, sess)
    db = PickleSession() if args.db == "pickle" else MongoSession()
    db.base_path = os.path.join(sess.index_name, "pickles") if db.db_type == "pickle" else f"experiments_{sess.index_name}"

    #metrics = ['silhouette']
    metrics = ['dbcv']
    #metrics = ['dbi']

    reducer = dimensionality_reducer()

    for exp in db.available_experiments(args.x):
        os.makedirs(f"{exp}", exist_ok=True)
        db.sub_path = exp
        for metric in metrics:
            run_for_metric(metric, reducer)
        
    db.close()