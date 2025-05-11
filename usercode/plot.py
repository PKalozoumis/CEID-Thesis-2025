import sys
import os
sys.path.append(os.path.abspath(".."))

from rich.console import Console
from functools import partial

from sentence_transformers import SentenceTransformer

from mypackage.elastic import ElasticDocument, Session
from mypackage.sentence import doc_to_sentences, iterative_merge, buggy_merge
from mypackage.clustering import chain_clustering, visualize_clustering
import pickle
from collections import namedtuple
from multiprocessing import Process, set_start_method
import argparse

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes


#set_start_method('spawn', force=True)

ProcessedDocument = namedtuple("ProcessedDocument", ["doc", "chains", "labels", "clusters"])
console = Console()

#=============================================================================================================

def full(pkl, imgpath):
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
        ax[i].set_title(f"{i:02}: Doc {p.doc.id:02}")
        
    fig.legend(handles=max_legend, loc='upper left', bbox_to_anchor=(pos.x0 + 0.05, pos.y0 + pos.height), ncols=3, prop=FontProperties(size=14), columnspacing=5)
    fig.savefig(os.path.join(imgpath, "full.png"))

#=============================================================================================================

def compare(pkl, imgpath):
    #This will compare the various methods
    pass

#=============================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-docs", action="store", default=None)
    parser.add_argument("name", nargs="?", action="store", type=str, default="default", help="Experiment name. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("-p", action="store", type=str, help="The type of plot to make", choices=[
        "full"
    ], default="full")

    args = parser.parse_args()

    func = {
        'full': full,
        'compare': compare
    }

    #---------------------------------------------------------------------------

    console.print(f"Running experiment '{args.name}'")

    if not args.docs:
        docs_to_retrieve = [1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 6415]
    else:
        docs_to_retrieve = args.docs.split(",")
    
    #Load pickles
    #---------------------------------------------------------------------------
    pkl = []

    for fname in map(lambda x: os.path.join("pickles", args.name, f"{x}.pkl"), docs_to_retrieve):
        with open(fname, "rb") as f:
            pkl.append(pickle.load(f))

    imgpath = os.path.join("images", args.name)

    os.makedirs(imgpath, exist_ok=True)


    #Run function
    #---------------------------------------------------------------------------
    func[args.p](pkl, imgpath)