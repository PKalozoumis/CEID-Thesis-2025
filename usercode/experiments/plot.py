import sys
import os
sys.path.append(os.path.abspath("../.."))

from rich.console import Console
from functools import partial

from sentence_transformers import SentenceTransformer

from mypackage.elastic import ElasticDocument, Session
from mypackage.sentence import doc_to_sentences, iterative_merge, buggy_merge
from mypackage.clustering import visualize_clustering
import pickle
from collections import namedtuple
from multiprocessing import Process, set_start_method
import argparse
import json

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes

import numpy as np
from helper import load_experiment, load_pickles
import math

#set_start_method('spawn', force=True)
console = Console()

#=============================================================================================================

def full(pkl, imgpath, experiment, index_name):
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
        ax[i].set_title(f"{i:02}: Doc {int(p.doc.id):02} ({index_name})")
        
    fig.legend(handles=max_legend, loc='upper left', bbox_to_anchor=(pos.x0 + 0.05, pos.y0 + pos.height), ncols=3, prop=FontProperties(size=14), columnspacing=5)
    fig.savefig(os.path.join(imgpath, f"full_{index_name.replace('-', '_')}_{experiment}.png"))

#=============================================================================================================

def compare(experiments, imgpath, docs, index_name):

    for i, doc in enumerate(docs):
        #Determine grid size
        N = len(experiments)
        b = math.ceil(math.sqrt(N))
        a = math.ceil(N / b) 

        fig, ax = plt.subplots(a, b, figsize=(b*4.33, a*4))
        
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.83, wspace=0.05, hspace=0.1)
        axes = ax.reshape((1, -1))[0]
        console.print(f"Plotting document {doc}")
        axes: list[Axes]

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        for ax, exp in zip(axes, experiments):
            exp_params = load_experiment(exp)
            pkl = load_pickles(exp, doc, index_name)
            visualize_clustering(pkl.chains, pkl.labels, ax=ax, return_legend=True, min_dista=exp_params['min_dista'])

            #Load experiment to get its title
            ax.set_title(exp_params['title'])

        fig.suptitle(f"Comparisons for Document {doc} ({index_name})")
        fig.savefig(os.path.join(imgpath, f"compare_{index_name.replace('-', '_')}_{i:02}_{doc}.png"))
        plt.close(fig)

#=============================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-docs", action="store", default=None)
    parser.add_argument("-i", action="store", type=str, default="pubmed-index", help="Index name", choices=[
        "pubmed-index",
        "arxiv-index"
    ])
    parser.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Experiment name. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("p", action="store", type=str, help="The type of plot to make", choices=[
        "full",
        "compare"
    ], default="full")

    args = parser.parse_args()

    #---------------------------------------------------------------------------

    console.print(f"Making plot: '{args.p}'")

    if not args.docs:
        if args.i == "pubmed-index":
            docs_to_retrieve = [1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 6415]
        elif args.i == "arxiv-index":
            docs_to_retrieve = list(range(10))
    else:
        docs_to_retrieve = args.docs.split(",")

    console.print("Session info:")
    console.print({'index_name': args.i, 'docs': docs_to_retrieve})
    print()

    imgpath = os.path.join(args.i, "images", args.p)
    os.makedirs(imgpath, exist_ok=True)

    #---------------------------------------------------------------------------

    if args.p == 'full':
        console.print(f"Plotting experiment '{args.x}'")
        pkl = load_pickles(args.x, docs_to_retrieve, args.i)
        
        full(pkl, imgpath, args.x, args.i)

    elif args.p == 'compare':
        with open("experiments.json", "r") as f:
            experiments = list(json.load(f).keys())
        compare(experiments, imgpath, docs_to_retrieve, args.i)

        