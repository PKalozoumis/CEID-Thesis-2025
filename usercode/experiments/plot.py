import sys
import os
sys.path.append(os.path.abspath("../.."))

from rich.console import Console
from functools import partial

from sentence_transformers import SentenceTransformer

from mypackage.elastic import ElasticDocument, Session
from mypackage.clustering import visualize_clustering
from mypackage.clustering.metrics import clustering_metrics, VALID_METRICS
from mypackage.storage import load_pickles, ProcessedDocument
from mypackage.helper import DEVICE_EXCEPTION
import pickle
from collections import namedtuple
from multiprocessing import Process, set_start_method
import argparse
import json
import shutil

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.lines import Line2D

import numpy as np
from helper import experiment_wrapper, ARXIV_DOCS, PUBMED_DOCS, document_index, experiment_names_from_dir
from mypackage.storage import load_pickles
import math
import warnings

from rich.rule import Rule

#set_start_method('spawn', force=True)
console = Console()

#=============================================================================================================

def full(pkl: list[ProcessedDocument], imgpath: str, experiment_name: str, sess: Session, no_outliers):
    #Make the figure
    #------------------------------------------------------------
    fig = plt.figure(figsize=(19.2,10.8))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95, hspace=0.1, wspace=0)
    fig.suptitle(f"Document clusters for experiment '{experiment_name}'{' (no outliers)' if no_outliers else ''}")

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
        legend_elements = visualize_clustering(p.chains, p.labels, ax=ax[i], return_legend=True, no_outliers=no_outliers)
        if len(legend_elements) > len(max_legend):
            max_legend = legend_elements
        ax[i].set_title(f"{i:02}: Doc {p.doc.id:02} ({sess.index_name})")
        
    fig.legend(handles=max_legend, loc='upper left', bbox_to_anchor=(pos.x0 + 0.05, pos.y0 + pos.height), ncols=3, prop=FontProperties(size=14), columnspacing=5)
    fig.savefig(os.path.join(imgpath, f"full_{sess.index_name.replace('-index', '')}_{experiment_name}{'_no_outliers' if no_outliers else ''}.png"))
    plt.close(fig)

#=============================================================================================================

def compare(experiment_names: str, imgpath, docs: list[int], sess: Session, *, metric: str = None, no_outliers):

    for i, doc in enumerate(docs):
        experiment_list = experiment_names_from_dir(os.path.join(sess.index_name, "pickles"), experiment_names)

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

        for ax, experiment_name in zip(axes, experiment_list):
            pkl = load_pickles(sess, os.path.join(sess.index_name, "pickles", experiment_name), doc)
            visualize_clustering(pkl.chains, pkl.labels, ax=ax, return_legend=True, min_dista=pkl.params['min_dista'], no_outliers=no_outliers)

            #Load experiment to get its title
            score = f" ({clustering_metrics(pkl.clustering, print=False)[metric]['value']:.3f})" if metric is not None else ""
            ax.set_title(pkl.params['title'] + score)

        fig.suptitle(f"Comparisons for Document {doc} ({sess.index_name})")
        fig.savefig(os.path.join(imgpath, f"compare_{sess.index_name.replace('-index', '')}_{document_index(sess.index_name, doc):02}_{doc}.png"))
        plt.close(fig)

#=============================================================================================================

def interdoc(pkl_list: list[ProcessedDocument], imgpath, sess: Session, no_outliers: bool = False):
    fig, ax = plt.subplots(figsize=(19.2,10.8))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95, hspace=0.1, wspace=0)
    fig.suptitle(f"Document chains visualized on the same space ({sess.index_name})")
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        cmap = plt.cm.get_cmap("tab20").colors

    for i, pkl in enumerate(pkl_list):
        visualize_clustering(pkl.chains, [i]*len(pkl.chains), ax=ax, return_legend=True, no_outliers=no_outliers)
        legend_elements.append(Patch(facecolor=cmap[(2*i + int(i > 9))%20], label=f'Document {pkl.doc.id:04}'))

    ax.legend(handles=legend_elements)
    fig.savefig(os.path.join(imgpath, f"interdoc_{sess.index_name.replace('-index', '')}_{experiment_name}{'_no_outliers' if no_outliers else ''}.png"))
    plt.close(fig)

#=============================================================================================================

def interdoc2(pkl_list: list[ProcessedDocument], imgpath, sess: Session, no_outliers: bool = False):
    fig, ax = plt.subplots(figsize=(19.2,10.8))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95, hspace=0.1, wspace=0)
    fig.suptitle(f"Document chains visualized on the same space ({sess.index_name})")
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        cmap = plt.cm.get_cmap("tab20").colors

    shapes = ['o', 's', '^', 'v', 'D', '*', 'x', '+', '<', '>']

    for i, pkl in enumerate(pkl_list):
        visualize_clustering(pkl.chains, pkl.labels, ax=ax, return_legend=True, shape=shapes[i], no_outliers=no_outliers)
        legend_elements.append(Line2D([0],[0],marker=shapes[i],linestyle='None',markersize=8, label=f'Document {pkl.doc.id:04}'))

    ax.legend(handles=legend_elements)
    fig.savefig(os.path.join(imgpath, f"interdoc2_{sess.index_name.replace('-index', '')}_{experiment_name}{'_no_outliers' if no_outliers else ''}.png"))
    plt.close(fig)

#=============================================================================================================

def centroids(pkl_list: list[ProcessedDocument], imgpath, sess: Session, extra_vector = None):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    fig.subplots_adjust(right=0.7)
    fig.suptitle(f"Cluster centroids visualized on the same space ({sess.index_name})")
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        cmap = plt.cm.get_cmap("tab20").colors

    centroids = []
    labels = []
    for i, pkl in enumerate(pkl_list):
        for cluster in pkl.clustering:
            if cluster.label > -1:
                centroids.append(cluster) #The vector will be extracted inside the visualization function
                labels.append(i)
        legend_elements.append(Patch(facecolor=cmap[(2*i + int(i > 9))%20], label=f'Document {pkl.doc.id:04}'))

    visualize_clustering(centroids, labels, ax=ax, return_legend=True, extra_vector=extra_vector)

    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1))
    fig.savefig(os.path.join(imgpath, f"centroids_{sess.index_name.replace('-index', '')}_{experiment_name}.png"))
    plt.close(fig)


#=============================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", action="store", default=None)
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Index name", choices=[
        "pubmed",
        "arxiv",
        "both"
    ])
    parser.add_argument("-x", nargs="?", action="store", type=str, default=None, help="Experiment name. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("p", action="store", type=str, help="The type of plot to make", choices=[
        "full",
        "compare",
        "interdoc",
        "interdoc2",
        "centroids",
        "query"
    ], default="full")
    parser.add_argument("-metric", action="store", type=str, default=None, help="Calculate an optional metric for each plot", choices=VALID_METRICS)
    parser.add_argument("--clear", action="store_true", default=False, help="Delete previous plots from the folder")
    parser.add_argument("--no-outliers", action="store_true", default=False, help="Removes outliers from the visualization")

    args = parser.parse_args()

    #---------------------------------------------------------------------------------------

    if args.i == "both":
        indexes = ["pubmed-index", "arxiv-index"]
        if args.d is not None:
            raise DEVICE_EXCEPTION("THE DOCUMENTS MUST CHOOSE... TO EXIST IN BOTH, IT INVITES FRACTURE.")
    else:
        indexes = [args.i + "-index"]

    #---------------------------------------------------------------------------

    for index in indexes:
        console.print(f"\nRunning for index '{index}'")
        console.print(Rule())

        console.print(f"Making plot: '{args.p}'")

        if not args.d:
            if index == "pubmed-index":
                docs_to_retrieve = PUBMED_DOCS
            elif index == "arxiv-index":
                docs_to_retrieve = ARXIV_DOCS
        else:
            docs_to_retrieve = [int(x) for x in args.d.split(",")]

        #---------------------------------------------------------------------------

        console.print("Session info:")
        console.print({'index_name': index, 'docs': docs_to_retrieve})
        print()

        imgpath = os.path.join(index, "images", args.p)
        
        if args.clear and os.path.exists(imgpath):
            shutil.rmtree(imgpath)
        
        os.makedirs(imgpath, exist_ok=True)
        sess = Session(index, base_path="../..", cache_dir="../cache", use="cache")

        #---------------------------------------------------------------------------
        if args.p == 'full':
            if args.x is None:
                args.x = "default"

            for experiment_name in args.x.split(","):
                console.print(f"Plotting experiment '{experiment_name}'")
                pkl = load_pickles(sess, os.path.join(sess.index_name, "pickles", experiment_name), docs_to_retrieve)
                
                full(pkl, imgpath, experiment_name, sess, args.no_outliers)
                print()

        #---------------------------------------------------------------------------
        elif args.p == 'compare':
            if args.x is None:
                args.x = "all"
                
            compare(args.x, imgpath, docs_to_retrieve, sess, metric=args.metric, no_outliers=args.no_outliers)

        #---------------------------------------------------------------------------
        elif args.p in ['interdoc', 'interdoc2', 'centroids', 'query']:
            #For a specific experiment, draw all cluster from all document onto one plot

            if args.x is None:
                args.x = "default"

            for experiment_name in args.x.split(","):
                console.print(f"Plotting experiment '{experiment_name}'")
                pkl = load_pickles(sess, os.path.join(sess.index_name, "pickles", experiment_name), docs_to_retrieve)
                
                match args.p:
                    case "interdoc": interdoc(pkl, imgpath, sess, args.no_outliers)
                    case "interdoc2": interdoc2(pkl, imgpath, sess, args.no_outliers)
                    case "centroids": centroids(pkl, imgpath, sess)
                    case "query":
                        query_text = "What are the primary behaviours and lifestyle factors that contribute to childhood obesity"
                        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
                        query_vector = model.encode(query_text)

                        centroids(pkl, imgpath, sess, extra_vector=query_vector)