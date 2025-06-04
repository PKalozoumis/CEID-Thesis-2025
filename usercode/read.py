import os
import sys
sys.path.append(os.path.abspath(".."))

import argparse

from mypackage.elastic import Session, ElasticDocument
from mypackage.helper import panel_print

from rich.console import Console

console = Console()

#===============================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", type=str, help="The index name")
    parser.add_argument("-d", action="store", type=str, help="Comma-separated list of document IDs")
    parser.add_argument("--store", action="store_true", help="Store to cache")
    parser.add_argument("--no-print", action="store_true", help="Disable printing")

    args = parser.parse_args()
    print(args)

    os.makedirs("cache", exist_ok=True)
    sess = Session(args.i, base_path="..", cache_dir=("cache" if args.store else None))

    docs = [ElasticDocument(sess, id=id, text_path="article") for id in args.d.split(",")]
    for doc in docs:
        print(doc.id)
        doc.get()

        if args.no_print:
            continue
        else:
            panel_print(doc.text)