#Some documents are generating too many clusters, while others are just fine
#For example, I'm using this list:
#[1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 6415]
#Documents 4355, and 372 have too many clusters and many outliers
#What makes those documents different?

import os
import sys

sys.path.append(os.path.abspath("../.."))

from collections import namedtuple
from mypackage.elastic import Session, ElasticDocument
from mypackage.helper import NpEncoder, create_table
from mypackage.sentence.metrics import chain_metrics
from mypackage.clustering.metrics import clustering_metrics

import numpy as np

from helper import experiment_wrapper, ARXIV_DOCS, PUBMED_DOCS
from mypackage.storage import load_pickles, ProcessedDocument

import pickle
from itertools import chain
import argparse
import json

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.pretty import Pretty
from rich.padding import Padding
from rich.columns import Columns

from collections import defaultdict

console = Console()

#=================================================================================================================

def stats(chains):
    chain_lengths = [len(c) for c in chains]

    data = {}

    data['num_chains'] = len(chains)
    data['avg_chain_length'] = np.average(chain_lengths)
    data['min_chain_length'] = np.min(chain_lengths)
    data['max_chain_length'] = np.max(chain_lengths)

    sentence_lengths = [len(c) for c in chain.from_iterable(chains)]
    data['num_sentences'] = len(sentence_lengths)
    data['num_words'] = np.sum(sentence_lengths)
    data['avg_sentence_length'] = np.average(sentence_lengths)
    data['min_sentence_length'] = np.min(sentence_lengths)
    data['max_sentence_length'] = np.max(sentence_lengths)

    return data

#=================================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-docs", action="store", type=str, default=None, help="Comma-separated list of docs")
    parser.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Index name", choices=[
        "pubmed",
        "arxiv"
    ]),
    parser.add_argument("mode", nargs="?", action="store", type=str, help="The type of plot to make", choices=[
        "single", #Run each experiment on the list on its own
        "compare"  #Compare the experiments side by side
    ], default="single")
    args = parser.parse_args()

    args.i += "-index"

    #-------------------------------------------------------------------------------------------

    if not args.docs:
        if args.i == "pubmed-index":
            docs_to_retrieve = PUBMED_DOCS
        elif args.i == "arxiv-index":
            docs_to_retrieve = ARXIV_DOCS
    else:
        docs_to_retrieve = [int(x) for x in args.docs.split(",")]

    #-------------------------------------------------------------------------------------------
    os.makedirs(os.path.join(args.i, "stats"), exist_ok=True)
    sess = Session(args.i, base_path="../..", cache_dir="../cache", use="cache")

    if args.mode == "single":
        for experiment in experiment_wrapper(args.x.split(','), must_exist=True, strict_iterable=True):
            rich_group_items = []
            rich_group_items.append(Pretty(experiment))

            pkl = load_pickles(sess, os.path.join(sess.index_name, "pickles", experiment['name']), docs_to_retrieve)
            out = []

            for i, p in enumerate(pkl):
                rich_group_items.append(Padding(Rule(f"{i:02}: Document {p.doc.id:04}", style="green"), (1,0)))

                #Chaining metrics
                rich_group_items.append(Padding(chain_metrics(p.chains, return_renderable=True)[1], (0,0,1,0)))

                #Clustering metrics
                rich_group_items.append(Padding(clustering_metrics(p.chains, p.labels, return_renderable=True)[1], (0,0,1,0)))

                #Stats
                data = {'id':docs_to_retrieve[i], 'index': i} | stats(p.chains)
                rich_group_items.append(Padding(create_table(['Statistic', 'Value'], data, title="Stats"), (0,0,1,0)))
                out.append(data)

            #Write to file
            with open(os.path.join(args.i, "stats", f"{experiment['name']}.json"), "w") as f:
                json.dump(out, f, cls=NpEncoder, indent="\t")

            console.print(Padding(Panel(Group(*rich_group_items), title=f"THIS NEXT EXPERIMENT: {experiment['name']}", border_style="green", highlight=True), (0,0,10,0)))


    #==========================================================================================================================

    elif args.mode == "compare":
        experiments = experiment_wrapper(args.x.split(','))

        if len(experiments) < 2:
            raise Exception("I SEE, YOU INTEND FOR NO CORRELATION")
        
        if len(experiments) > 2:
            raise Exception("IT SPILLS BEYOND THE BRINK OF THE DEVICE")
        
        #We iterate over all the documents
        #For each doc, we want to run and compare the selected experiments
        #Only then do we move on to the next document
        for i, doc in enumerate(docs_to_retrieve):

            chain_rows = defaultdict(list)
            cluster_rows = defaultdict(list)
            stat_rows = defaultdict(list)
            column_names = []

            #Iterate over the experiments
            for xp_num, xp in enumerate(experiments):
                rich_group_items = []

                #Open only one experiment
                pkl = load_pickles(sess, os.path.join(sess.index_name, "pickles", xp['name']), doc)

                #Add new column (for new experiment) to the table data
                #-----------------------------------------------------------------------
                column_names.append(xp['name'])

                for temp in chain_metrics(pkl.chains).values():
                    chain_rows[temp['name']].append(temp['value'])

                for temp in clustering_metrics(pkl.chains, pkl.labels).values():
                    cluster_rows[temp['name']].append(temp['value'])

                for k,v in ({'id':doc, 'index': i} | stats(pkl.chains)).items():
                    stat_rows[k].append(v)

                #-----------------------------------------------------------------------

                rich_group_items.append(Padding(create_table(['Metric', 'Value'], data, title="Stats"), (0,0,1,0)))
                rich_group_items.append(Padding(create_table(['Metric', 'Value'], data, title="Stats"), (0,0,1,0)))
                rich_group_items.append(Padding(create_table(['Statistic', 'Value'], data, title="Stats"), (0,0,1,0)))

                cols.append(Group(*rich_group_items))

        console.print(Columns(cols))



