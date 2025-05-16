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
from mypackage.helper import NpEncoder
from mypackage.sentence.metrics import chain_metrics

import numpy as np

from helper import experiment_wrapper
from mypackage.storage import load_pickles, ProcessedDocument

import pickle
from rich.console import Console
from itertools import chain
import argparse
import json

console = Console()

#=================================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-docs", action="store", type=str, default=None, help="Comma-separated list of docs")
    parser.add_argument("name", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("-i", action="store", type=str, default="pubmed-index", help="Index name", choices=[
        "pubmed-index",
        "arxiv-index"
    ])
    args = parser.parse_args()

    #-------------------------------------------------------------------------------------------

    if not args.docs:
        if args.i == "pubmed-index":
            docs_to_retrieve = [1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 6415]
        elif args.i == "arxiv-index":
            docs_to_retrieve = list(range(10))
    else:
        docs_to_retrieve = [int(x) for x in args.docs.split(",")]

    #-------------------------------------------------------------------------------------------
    os.makedirs(os.path.join(args.i, "stats"), exist_ok=True)
    sess = Session(args.i, base_path="../..", cache_dir="../cache", use="cache")

    for experiment in experiment_wrapper(args.name.split(','), True, strict_iterable=True):
        if experiment is None:
            break

        console.print(f"THIS NEXT EXPERIMENT:")
        console.print(experiment)
        pkl = load_pickles(sess, os.path.join(sess.index_name, "pickles", experiment['name']), docs_to_retrieve)

        out = []

        for i, p in enumerate(pkl):
            print(type(p))
            chain_metrics(p.chains)
            data = {'id':docs_to_retrieve[i], 'index': i}

            chain_lengths = [len(c) for c in p.chains]
            data['num_chains'] = len(p.chains)
            data['avg_chain_length'] = np.average(chain_lengths)
            data['min_chain_length'] = np.min(chain_lengths)
            data['max_chain_length'] = np.max(chain_lengths)

            sentence_lengths = [len(c) for c in chain.from_iterable(p.chains)]
            data['num_sentences'] = len(sentence_lengths)
            data['num_words'] = np.sum(sentence_lengths)
            data['avg_sentence_length'] = np.average(sentence_lengths)
            data['min_sentence_length'] = np.min(sentence_lengths)
            data['max_sentence_length'] = np.max(sentence_lengths)

            console.print(data)
            out.append(data)
        
        console.print("VERY, VERY INTERESTING")

        with open(os.path.join(args.i, "stats", f"{experiment['name']}.json"), "w") as f:
            json.dump(out, f, cls=NpEncoder, indent="\t")


        



