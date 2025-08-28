import sys
import os
sys.path.append(os.path.abspath("../.."))

import argparse

#ARGUMENT PARSING
#=============================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", action="store", type=str, help="Operation to perform", choices = [
        "read", #Reads the parameters (experiment) used for a specific pickle file
        "clear-unused", #Removes unused experiments from the specified index
        "clear-temp", #Removes temporary experiments from the specified index
        "list-unused",
        "list-temp"
    ])
    parser.add_argument("-d", action="store", type=str, default=None, help="Comma-separated list of docs. Leave blank for a predefined set of test documents. -1 for all")
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Comma-separated list of index names")
    parser.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("-db", action="store", type=str, default='mongo', help="Database to load the preprocessing results from", choices=['mongo', 'pickle'])
    parser.add_argument("--cache", action="store_true", default=False, help="Retrieve docs from cache instead of elasticsearch")
    args = parser.parse_args()

#IMPORTS
#=============================================================================================================
from rich.console import Console
from mypackage.elastic import Session       
from mypackage.helper import panel_print
from mypackage.experiments import ExperimentManager
from mypackage.storage import DatabaseSession, MongoSession, PickleSession
from rich.pretty import Pretty
from rich.rule import Rule
import shutil
from mypackage.helper import DEVICE_EXCEPTION

console = Console()

#=============================================================================================================

if __name__ == "__main__":
    indexes = args.i.split(",")
    exp_manager = ExperimentManager("../common/experiments.json")

    if len(indexes) > 1:
        if args.d is not None:
            raise DEVICE_EXCEPTION("THE DOCUMENTS MUST CHOOSE... TO EXIST IN ALL, IT INVITES FRACTURE.")

    #-------------------------------------------------------------------------------------------

    for index in indexes:

        console.print(f"\nRunning '{args.mode}' for index '{index}'")
        console.print(Rule())

        sess = Session(index, base_path="../common", cache_dir="../cache", use="cache" if args.cache else "client")
        docs = exp_manager.get_docs(args.d, sess)

        if args.db == "pickle":
            db = PickleSession(os.path.join(index, "pickles"))
        else:
            db = MongoSession(db_name=f"experiments_{index}")

        #-------------------------------------------------------------------------------------------

        #Read experiment parameters
        if args.mode == "read":
            
            for THIS_NEXT_EXPERIMENT in db.available_experiments(args.x):
                db.sub_path = THIS_NEXT_EXPERIMENT
                for i, doc in enumerate(docs):
                    title = f"{index} -> {THIS_NEXT_EXPERIMENT} -> {exp_manager.document_index(index, doc.id, i):02}: Document {doc.id:04}"
                    pkl = db.load(sess, doc)
                    panel_print(Pretty(pkl.params), title)

        #-------------------------------------------------------------------------------------------

        elif args.mode in ["clear-unused", "clear-temp"]:
            experiment_names = exp_manager.experiment_names()

            removed = False
            for dir in db.list_experiments():
                db.sub_path = dir
                if dir not in experiment_names:
                    is_temp = db.is_temp()
                    if (args.mode == "clear-unused" and not is_temp) or (args.mode == "clear-temp" and is_temp):
                        removed = True
                        db.delete()
                        console.print(f"Removed {dir}")
                
            if removed:
                console.print("\nIT'S AS IF THEY WERE NEVER THERE AT ALL\n")
            else:
                console.print("THEY HAVE ALREADY BEEN CAST INTO THE DEPTHS\n")

        #-------------------------------------------------------------------------------------------

        elif args.mode in ["list-unused", "list-temp"]:
            experiment_names = exp_manager.experiment_names()

            found = False
            for dir in db.list_experiments():
                db.sub_path = dir
                if dir not in experiment_names:
                    is_temp = db.is_temp()
                    if (args.mode == "list-unused" and not is_temp) or (args.mode == "list-temp" and is_temp):
                        found = True
                        console.print(f"{dir}")
                
            if not found:
                console.print("NOTHING BUT SILENCE\n")


        db.close()