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
    args = parser.parse_args()

#=================================================================================================================

from collections import namedtuple
from mypackage.elastic import Session, ElasticDocument
from mypackage.helper import NpEncoder, create_table, write_to_excel_tab, DEVICE_EXCEPTION
from mypackage.sentence.metrics import chain_metrics, within_chain_similarity_at_k
from mypackage.clustering.metrics import clustering_metrics, stats
from mypackage.sentence import SentenceChain
from mypackage.clustering import ChainClustering

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

console = Console()

#=================================================================================================================

def similarity_at_k():
    dframes=[]
    for exp in db.available_experiments(args.x):
        db.sub_path = exp
        processed = db.load(sess, docs, skip_missing_docs=True)    

        df = within_chain_similarity_at_k([d.chains for d in processed if d is not None])
        df.columns = [exp]
        print(df)
        dframes.append(df)

    #I cannot draw merged df due to the line breaking at NaN values
    #I have to draw each df separately instead
    cmap = plt.cm.get_cmap("tab10").colors
    for i, df in enumerate(dframes):
        plt.plot(df.index, df.to_numpy(), marker='o', linestyle='-', label=df.columns)
        db._sub_path = df.columns[0]
        params = db.get_experiment_params()
        plt.axhline(y=params['threshold'], color=cmap[i], linestyle='--')
    plt.legend()
    plt.show(block=True)

#=================================================================================================================

def chaining():

    experiment_chains = []

    for exp in db.available_experiments(args.x):
        db.sub_path = exp
        processed = db.load(sess, docs, skip_missing_docs=True)    
        experiment_chains.append(list(chain.from_iterable(proc.chains for proc in processed)))

    sizes_per_experiment = []

    for chains in experiment_chains:
        sizes_per_experiment.append([len(c) for c in chains])

    df = pd.DataFrame({
        'Size': list(chain.from_iterable(sizes_per_experiment)),
        'Experiment': [f'exp{i}' for i, exp in enumerate(sizes_per_experiment) for _ in exp]
    })
    
    # Count number of groups per size for each method
    counts = df.groupby(['Experiment', 'Size']).size().unstack(fill_value=0)
    # Remove sizes that are zero in all experiments
    counts = counts.loc[:, (counts != 0).any(axis=0)]

    # Convert to percentages
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    # Plot bar plot
    ax = percentages.T.plot(kind='bar', width=0.7)  # .T so sizes on x-axis
    ax.set_xlabel('Group Size')
    ax.set_ylabel('Percentage of Groups')
    ax.set_title('Group Size Distribution by Method')
    plt.xticks(rotation=0)  # keep x-ticks horizontal
    plt.show()


#=========================================================================================================

if __name__ == "__main__":
    exp_manager = ExperimentManager("../common/experiments.json")
    db = PickleSession() if args.db == "pickle" else MongoSession()
    sess = Session(args.index, base_path="../common", cache_dir="../cache", use="cache" if args.cache else "client")
    docs = exp_manager.get_docs(args.d, sess)    
    db.base_path = os.path.join(sess.index_name, "pickles") if db.db_type == "pickle" else f"experiments_{sess.index_name}"
    
    similarity_at_k()
        
    db.close()