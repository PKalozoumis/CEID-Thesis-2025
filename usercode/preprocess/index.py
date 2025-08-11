
import sys
import os
sys.path.append(os.path.abspath(".."))

import argparse
import time

from rich.progress import Progress
from rich.console import Console

from mypackage.elastic import elasticsearch_client
from mypackage.elastic.index import empty_index, create_index
from mypackage.helper.collection_helper import generate_examples, to_bulk_format
from mypackage.helper import line_count, batched, DEVICE_EXCEPTION

console = Console()

#=========================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Elasticsearch Index Management")
    parser.add_argument("operation", nargs="?", action="store", type=str, default="index", choices=[
        "index",
        "empty"
    ], help="Operation to perform")
    parser.add_argument("-i", action="store", type=str, default=None, help="Index name")
    parser.add_argument("-doc-limit", action="store", type=int, default=None, help="Only index a certain number of docs")
    parser.add_argument("--remove-duplicates", action="store_true", default=False, help="Remove duplicate sentences from each doc")
    args = parser.parse_args()

    if args.name is None:
        raise DEVICE_EXCEPTION("DOES IT NOT HAVE A NAME?")

    #Paths
    #---------------------------------------------
    index_name = args.name
    dataset_path = "../collection/pubmed.txt"
    credentials_path = "credentials.json"
    cert_path = "http_ca.crt"
    mapping_path = "mapping.json"
    #---------------------------------------------

    client = elasticsearch_client(credentials_path, cert_path)

    if args.operation == "empty":
        print(f"Emptying index {index_name}...")
        empty_index(client, index_name)
    
    elif args.operation == "index":
        console.print(f"[green]WE CALLED IT \"{args.name}\"[green]\n")
        num_docs = line_count(dataset_path)

        #When indexind document using bulk queries, the docs will be split into batches
        #This is because Elasticsearch had a 100MB limit on query body (roughly 1500 docs)
        #Ideally, the bulk queries would happen in parallel...
        batch_size = 1500

        #Add docs to elasticsearch
        create_index(client, index_name, mapping_path)

        t = time.time()
        bulk = to_bulk_format(generate_examples(dataset_path, doc_limit=args.doc_limit, remove_duplicates=args.remove_duplicates))
        batches = batched(bulk, 2*batch_size)

        with Progress() as progress:
            task = progress.add_task("[green]Indexing documents...", total=num_docs)

            #Add to elasticsearch
            for batch in batches:
                client.bulk(index=index_name, operations=batch)
                progress.update(task, advance=batch_size)
            
        console.print("\n[green]PREPARATIONS ARE COMPLETE[/green]")
        print(f"\nElastic time: {round(time.time() - t, 2)}s")