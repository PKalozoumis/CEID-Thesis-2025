#How does chaining affect the clustering result

import sys
import os
sys.path.append(os.path.abspath("../.."))

from rich.console import Console
from mypackage.elastic import Session
import argparse
from mypackage.helper import panel_print
from helper import CHOSEN_DOCS, document_index, all_experiments, experiment_names_from_dir
from mypackage.storage import load_pickles
from rich.pretty import Pretty
from rich.rule import Rule
import shutil
from mypackage.helper import DEVICE_EXCEPTION

console = Console()

#=============================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", action="store", type=str, help="Operation to perform", choices = [
        "read", #Reads the parameters (experiment) used for a specific pickle file
        "clear", #Removes unused experiments from the specified index
        "clear-temp", #Removes temporary experiments from the specified index
        "list-unused",
        "list-temp"
    ])
    parser.add_argument("-d", action="store", type=str, default=None, help="Comma-separated list of docs")
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Comma-separated list of index names")
    parser.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    args = parser.parse_args()

    indexes = args.i.split(",")

    if len(indexes) > 1:
        if args.d is not None:
            raise DEVICE_EXCEPTION("THE DOCUMENTS MUST CHOOSE... TO EXIST IN ALL, IT INVITES FRACTURE.")

    #-------------------------------------------------------------------------------------------

    for index in indexes:

        console.print(f"\nRunning '{args.mode}' for index '{index}'")
        console.print(Rule())

        if not args.d:
            docs_to_retrieve = CHOSEN_DOCS.get(index, list(range(10)))
        else:
            docs_to_retrieve = [int(x) for x in args.d.split(",")]

        sess = Session(index, use="cache", cache_dir="../cache")

        #-------------------------------------------------------------------------------------------

        if args.mode == "read":
            base_path = os.path.join(index, "pickles")
            
            for THIS_NEXT_EXPERIMENT in experiment_names_from_dir(base_path, args.x):
                for i, doc in enumerate(docs_to_retrieve):
                    title = f"{index} -> {THIS_NEXT_EXPERIMENT} -> {document_index(index, doc, i):02}: Document {doc:04}"
                    pkl = load_pickles(sess, os.path.join(index, "pickles", THIS_NEXT_EXPERIMENT), doc)
                    panel_print(Pretty(pkl.params), title)

        #-------------------------------------------------------------------------------------------

        elif args.mode in ["clear", "clear-temp"]:
            base_path = os.path.join(index, "pickles")
            experiment_names = set(all_experiments(names_only=True))

            removed = False
            for dir in os.listdir(base_path):
                exp_path = os.path.join(base_path, dir)
                if dir not in experiment_names:

                    is_temp = os.path.exists(os.path.join(exp_path, ".temp"))

                    if (args.mode == "clear" and not is_temp) or (args.mode == "clear-temp" and is_temp):
                        removed = True
                        shutil.rmtree(exp_path)
                        console.print(f"Removed {dir}")
                
            if removed:
                console.print("\nIT'S AS IF THEY WERE NEVER THERE AT ALL\n")
            else:
                console.print("THEY HAVE ALREADY BEEN CAST INTO THE DEPTHS\n")

        #-------------------------------------------------------------------------------------------

        elif args.mode in ["list-unused", "list-temp"]:
            base_path = os.path.join(index, "pickles")
            experiment_names = set(all_experiments(names_only=True))

            found = False
            for dir in os.listdir(base_path):
                exp_path = os.path.join(base_path, dir)
                if dir not in experiment_names:

                    is_temp = os.path.exists(os.path.join(exp_path, ".temp"))

                    if (args.mode == "list-unused" and not is_temp) or (args.mode == "list-temp" and is_temp):
                        found = True
                        console.print(f"{dir}")
                
            if not found:
                console.print("NOTHING BUT SILENCE\n")