import sys
import os
sys.path.append(os.path.abspath("../.."))

from rich.console import Console
from functools import partial

from sentence_transformers import SentenceTransformer

from mypackage.elastic import ElasticDocument, Session
from mypackage.sentence import doc_to_sentences, iterative_merge, buggy_merge
from mypackage.clustering import visualize_clustering
from mypackage.clustering.metrics import clustering_metrics, VALID_METRICS
from mypackage.storage import load_pickles
import pickle
from collections import namedtuple
from multiprocessing import Process, set_start_method
import argparse
import json
import shutil

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes

import numpy as np
from helper import load_experiments, experiment_wrapper, ARXIV_DOCS, PUBMED_DOCS, document_index
from mypackage.storage import load_pickles
import math

#set_start_method('spawn', force=True)
console = Console()

#=============================================================================================================

def full(pkl, imgpath, experiment, sess: Session):
    #Make the figure
    #------------------------------------------------------------
    fig = plt.figure(figsize=(19.2,10.8))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95, hspace=0.1, wspace=0)
    fig.suptitle("Plots")

    ax: list[list[Axes]] = []
    pos = [1,4,5,6,7,8,9,10,11,12]
    for i in range(10):
        ax.append(fig.add_subplot(3,4,pos[i]))
    
    temp = fig.add_subplot(3,4,2)
    pos = temp.get_position()
    fig.delaxes(temp)

    max_legend = []

    #Plottting
    #------------------------------------------------------------
    for i, p in enumerate(pkl):
        console.print(f"Plotting document {p.doc.id}")
        legend_elements = visualize_clustering(p.chains, p.labels, ax=ax[i], return_legend=True)
        if len(legend_elements) > len(max_legend):
            max_legend = legend_elements
        ax[i].set_title(f"{i:02}: Doc {p.doc.id:02} ({sess.index_name})")
        
    fig.legend(handles=max_legend, loc='upper left', bbox_to_anchor=(pos.x0 + 0.05, pos.y0 + pos.height), ncols=3, prop=FontProperties(size=14), columnspacing=5)
    fig.savefig(os.path.join(imgpath, f"full_{sess.index_name.replace('-index', '')}_{experiment}.png"))

#=============================================================================================================

def compare(experiment_names: str|list[str], imgpath, docs: list[int], sess: Session, *, metric: str = None):

    for i, doc in enumerate(docs):
        experiment_list = experiment_wrapper(experiment_names)

        #Determine grid size
        N = len(experiment_list)
        b = math.ceil(math.sqrt(N))
        a = math.ceil(N / b)

        fig, ax_grid = plt.subplots(a, b, figsize=(b*4.33, a*4))
        
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.83, wspace=0.05, hspace=0.1)

        axes: list[Axes]
        if isinstance(ax_grid, np.ndarray):
            axes = ax_grid.reshape((1, -1))[0]
        else:
            axes = [ax_grid]
        
        console.print(f"Plotting document {doc}")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        for ax, exp_params in zip(axes, experiment_list):
            pkl = load_pickles(sess, os.path.join(sess.index_name, "pickles", exp_params['name']), doc)
            visualize_clustering(pkl.chains, pkl.labels, ax=ax, return_legend=True, min_dista=exp_params['min_dista'])

            #Load experiment to get its title
            score = f" ({clustering_metrics(pkl.chains, pkl.labels, print=False)[metric]['value']:.3f})" if metric is not None else ""
            ax.set_title(exp_params['title'] + score)

        fig.suptitle(f"Comparisons for Document {doc} ({sess.index_name})")
        fig.savefig(os.path.join(imgpath, f"compare_{sess.index_name.replace('-index', '')}_{document_index(sess.index_name, doc):02}_{doc}.png"))
        plt.close(fig)

#=============================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-docs", action="store", default=None)
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Index name", choices=[
        "pubmed",
        "arxiv"
    ])
    parser.add_argument("-x", nargs="?", action="store", type=str, default=None, help="Experiment name. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("p", action="store", type=str, help="The type of plot to make", choices=[
        "full",
        "compare"
    ], default="full")
    parser.add_argument("-metric", action="store", type=str, default=None, help="Calculate an optional metric for each plot", choices=VALID_METRICS)

    args = parser.parse_args()

    args.i += "-index"

    #---------------------------------------------------------------------------
    console.print(f"Making plot: '{args.p}'")

    if not args.docs:
        if args.i == "pubmed-index":
            docs_to_retrieve = PUBMED_DOCS
        elif args.i == "arxiv-index":
            docs_to_retrieve = ARXIV_DOCS
    else:
        docs_to_retrieve = [int(x) for x in args.docs.split(",")]

    console.print("Session info:")
    console.print({'index_name': args.i, 'docs': docs_to_retrieve})
    print()

    imgpath = os.path.join(args.i, "images", args.p)
    
    if os.path.exists(imgpath):
        shutil.rmtree(imgpath)
    
    os.makedirs(imgpath, exist_ok=True)
    #---------------------------------------------------------------------------
    sess = Session(args.i, base_path="../..", cache_dir="../cache", use="cache")

    if args.p == 'full':
        if args.x is None:
            args.x = "default"

        for experiment_name in args.x.split(","):
            console.print(f"Plotting experiment '{experiment_name}'")
            pkl = load_pickles(sess, os.path.join(sess.index_name, "pickles", experiment_name), docs_to_retrieve)
            
            full(pkl, imgpath, experiment_name, sess)
            print()

    elif args.p == 'compare':
        if args.x is None:
            args.x = "all"
             
        compare(args.x.split(","), imgpath, docs_to_retrieve, sess, metric=args.metric)

        