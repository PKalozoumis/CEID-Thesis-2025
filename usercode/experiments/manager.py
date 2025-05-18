#How does chaining affect the clustering result

import sys
import os
sys.path.append(os.path.abspath("../.."))

from rich.console import Console
from mypackage.elastic import Session
import argparse
from mypackage.helper import panel_print
from helper import experiment_wrapper, ARXIV_DOCS, PUBMED_DOCS, document_index, all_experiments
from mypackage.storage import load_pickles
from rich.pretty import Pretty
import shutil

console = Console()

#=============================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", action="store", type=str, help="Operation to perform", choices = [
        "read", #Reads the parameters (experiment) used for a specific pickle file
        "clean" #Removes unused experiments from the specified index
    ])
    parser.add_argument("-d", action="store", type=str, default=None, help="Comma-separated list of docs")
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Index name", choices=[
        "pubmed",
        "arxiv"
    ])
    parser.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    args = parser.parse_args()

    args.i += "-index"

    #-------------------------------------------------------------------------------------------

    if not args.d:
        if args.i == "pubmed-index":
            docs_to_retrieve = PUBMED_DOCS
        elif args.i == "arxiv-index":
            docs_to_retrieve = ARXIV_DOCS
    else:
        docs_to_retrieve = [int(x) for x in args.d.split(",")]

    #-------------------------------------------------------------------------------------------

    if args.mode == "read":
        for THIS_NEXT_EXPERIMENT in experiment_wrapper(args.x.split(','), strict_iterable=True, must_exist=True):
            for i, doc in enumerate(docs_to_retrieve):
                title = f"{args.i} -> {THIS_NEXT_EXPERIMENT['name']} -> {document_index(args.i, doc, i):02}: Document {doc:04}"
                pkl = load_pickles(Session(args.i, use="cache", cache_dir="../cache"), os.path.join(args.i, "pickles", THIS_NEXT_EXPERIMENT['name']), doc)

                panel_print(Pretty(pkl.params), title)

    elif args.mode == "clean":
        base_path = os.path.join(args.i, "pickles")
        experiment_names = set(all_experiments(names_only=True))

        removed = False
        for dir in os.listdir(base_path):
            if dir not in experiment_names:
                removed = True
                shutil.rmtree(os.path.join(base_path, dir))
                console.print(f"Removed {dir}")
            
        if removed:
            console.print("\nIT'S AS IF THEY WERE NEVER THERE AT ALL\n")
        else:
            console.print("\nTHEY HAVE ALREADY BEEN CAST INTO THE DEPTHS\n")
