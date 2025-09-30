'''
Caches sentence embeddings for all documents, so that further experiments can be compiled in 3 minutes instead of 20
Embeddings are only tied to the model. As long as that stays consistent, we good
'''

import sys
import os
sys.path.append(os.path.abspath("../.."))

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the preprocessing step for the specified documents, and with the specified parameters (experiments)")
    parser.add_argument("-d", action="store", type=str, default=None, help="Comma-separated list of docs or document range. Leave blank for a predefined set of test documents. -1 for all")
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Comma-separated list of index names")
    parser.add_argument("-m", "--model", action="store", type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Name or alias of the sentence embedding model to use")
    parser.add_argument("--cache", action="store_true", default=False, help="Retrieve docs from cache instead of elasticsearch")
    parser.add_argument("-nprocs", action="store", type=int, default=1, help="Number of processes")
    parser.add_argument("-dev", "--device", action="store", type=str, default="cpu", choices=["cpu", "gpu"], help="Device for the embedding model")
    parser.add_argument("--spawn", action="store_true", default=False, help="Set process start method to 'spawn'")
    parser.add_argument("-l", "--limit", action="store", type=int, default=None, help="Document limit for scrolling corpus")
    parser.add_argument("-b", "--batch-size", action="store", type=int, default=2000, help="Batch size for scrolling corpus")
    parser.add_argument("-st", "--scroll-time", action="store", type=str, default="1000s", help="Scroll time for scrolling corpus")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-w", "--warnings", action="store_true", default=False, help="Show warnings")

    args = parser.parse_args()

import traceback
import warnings
from sentence_transformers import SentenceTransformer
import pickle
from multiprocessing import Pool
import torch.multiprocessing as mp

from rich.console import Console
from rich.rule import Rule
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from mypackage.elastic import ElasticDocument, Session
from mypackage.sentence import doc_to_sentences, sentence_transformer_from_alias
from mypackage.helper import batched
from mypackage.experiments import ExperimentManager

if not args.warnings:
    warnings.filterwarnings("ignore")
console = Console()
db = None

#==============================================================================================

def initializer(i, a, m, cpu_model = None):
    global index_name, model, model_name, args
    model_name = m
    index_name = i
    args = a

    if cpu_model is None:
        #Initialize the model in each one of the pool's processes
        #This is the only way to use model in GPU, as sharing is not allowed
        model = SentenceTransformer(model_name, device='cuda')
    else:
        model = cpu_model

#==============================================================================================

def work(doc: ElasticDocument):
    global index_name, model, model_name, args

    if args.verbose: console.print(f"Document {doc.id:02}: Creating sentences...")
    sentences = doc_to_sentences(doc, model, sep="\n")
    if args.verbose: console.print(f"Document {doc.id:02}: Created {len(sentences)} sentences")
    
    #Save to disk
    base_path = os.path.join("embedding_cache", index_name, model_name.replace("/", "-"))
    os.makedirs(base_path, exist_ok=True)
    with open(os.path.join(base_path, f"{doc.id}.pkl"), "wb") as f:
        pickle.dump([s.vector for s in sentences], f)

#==============================================================================================

if __name__ == "__main__":
    exp_manager = ExperimentManager("../common/experiments.json")
    model_name = sentence_transformer_from_alias(args.model, "../common/model_aliases.json")

    for index in args.i.split(","):
        console.print(f"\nRunning for index '{index}'")
        console.print(Rule())

        sess = Session(index, base_path="../common", cache_dir="../cache", use= ("cache" if args.cache else "client"))
        docs = exp_manager.get_docs(args.d, sess, scroll_batch_size=args.batch_size, scroll_time=args.scroll_time, scroll_limit=args.limit)

        console.print("Session info:")
        console.print(f"DEVICE: {args.device}")
        console.print(f"Model: {args.model}")
        print()

        cpu_model = None
        if args.device == "cpu":
            cpu_model = SentenceTransformer(model_name, device='cpu')

        try:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                disable=args.verbose
            ) as progress:
                
                batch_size = args.batch_size if args.d == "-1" else len(docs)

                #Serial execution
                if args.nprocs == 1:
                    console.print("Serial execution")
                    initializer(index, args, model_name, cpu_model)

                    for i, batch in enumerate(batched(docs, batch_size)):
                        task = progress.add_task(f"Processing documents for batch {i}...", total=batch_size, start=True)

                        for doc in batch:
                            work(doc)
                            progress.update(task, advance=1)

                #Parallel execution with pool
                else:
                    console.print(f"Starting multiprocessing pool with {args.nprocs} processes\n")
                    if args.spawn:
                        mp.set_start_method('spawn', force=True)
                    try:
                        with Pool(processes=args.nprocs, initializer=initializer, initargs=(index, args, model_name, cpu_model)) as pool:
                            
                            #Receive a batch of documents from Elasticsearch
                            #Pass the workload to the pool
                            #The workload should be higher than the pool size
                            for i, batch in enumerate(batched(docs, args.batch_size)):
                                workload = []

                                #Retrieve documents
                                #Remove the unserializeable session object from them
                                for doc in batch:
                                    try:
                                        doc.get()
                                        doc.session = None
                                        workload.append(doc)
                                    except Exception:
                                        continue

                                task = progress.add_task(f"Processing documents for batch {i}...", total=len(workload), start=True)

                                #Distribute work
                                for res in pool.imap_unordered(work, workload):
                                    progress.update(task, advance=1)
                    except BaseException as e:
                        pool.terminate()
                        pool.join()
                        raise e
        except Exception as e:
            traceback.print_exc()
        except KeyboardInterrupt:
            pass
        finally:
            continue