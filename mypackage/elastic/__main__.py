
import sys
import os
from multiprocessing import Process, Pool
from .elastic import elasticsearch_client
from .index import empty_index, create_index
from ..helper.collection_helper import generate_examples, to_bulk_format
import argparse
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import time
from more_itertools import divide
import glob
from rich.progress import Progress
from ..helper import line_count, file_batch, batched

#=========================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Elasticsearch Index Management")
    parser.add_argument("--empty", action="store_true", help="Empty index")
    parser.add_argument("--dict", action="store_true", help="Create dictionary", default=False)
    parser.add_argument("--phrase", action="store_true", help="Train phrase model", default=False)
    parser.add_argument("--no-index", action="store_true", help="Index documents", default=False)
    parser.add_argument("-doc-limit", action="store", type=int, default=None)
    parser.add_argument("-nprocs", action="store", type=int, default=1, help="Number of processes for dictionary")
    args = parser.parse_args()

    #Paths
    #---------------------------------------------
    index_name = "pubmed-index"
    collection_path = "collection"
    dataset_file_name = "pubmed.txt"
    models_path = "models"
    #---------------------------------------------

    if not args.no_index:
        client = elasticsearch_client()
    
    task_flags = [args.empty, not args.no_index, args.phrase, args.dict]
    tasks = ["Emptying Index", "Indexing", "Phrase Model", f"Dictionary ({args.nprocs} processes)"]
    filtered = [task for task, flag in zip(tasks, task_flags) if flag]
    print(f"Tasks: {', '.join(filtered)}")

    if args.empty and (args.dict or args.phrase or args.doc_limit):
        print("You cannot have --empty with other actions")
        sys.exit()

    proceed = input("Proceed (y/n)?: ")

    if proceed not in ["y", "yes"]:
        print("Exiting...")
        sys.exit()

    print()

    if args.empty:
        print(f"Emptying index {index_name}...")
        empty_index(client, index_name)
    else:
        file_path = os.path.join(collection_path, dataset_file_name)
        num_docs = line_count(file_path)

        #When indexind document using bulk queries, the docs will be split into batches
        #This is because Elasticsearch had a 100MB limit on query bod (roughly 1500 docs)
        #Ideally, the bulk queries would happen in parallel...
        batch_size = 1500

        if args.dict and not os.path.exists(os.path.join(models_path, "phrase_model.pkl")):
            print("Cannot train dictionary, because there is not an available phrase model")
            sys.exit()

        #WORK FUNCTIONS
        #===================================================================================

        def train_phrase_model():
            t = time.time()
            stopwords = frozenset(set(STOPWORDS) - set(ENGLISH_CONNECTOR_WORDS))
            tokenized_docs = map(lambda doc: simple_preprocess(doc['summary']), generate_examples(file_path, doc_limit=args.doc_limit))
            filtered_docs = map(lambda doc: [token for token in doc if token not in stopwords], tokenized_docs)
            phrase_model = Phrases(filtered_docs, 6, 15, connector_words=ENGLISH_CONNECTOR_WORDS)
            print(f"Phrase time: {round(time.time() - t, 2)}s")

            phrase_model.save(os.path.join(models_path, "phrase_model.pkl"))

        #------------------------------------------------------------------------------------------

        def indexing():
            t = time.time()
            bulk = to_bulk_format(generate_examples(file_path, doc_limit=args.doc_limit))
            batches = batched(bulk, 2*batch_size)

            with Progress() as progress:
                task = progress.add_task("[green]Indexing documents...", total=num_docs)

                #Add to elasticsearch
                for batch in batches:
                    client.bulk(index=index_name, operations=batch)
                    progress.update(task, advance=batch_size)
                
            print(f"Elastic time: {round(time.time() - t, 2)}s")

        #------------------------------------------------------------------------------------------

        def train_dictionary(id, offsets):
            t = time.time()
            offsets = list(offsets)

            dic = Dictionary()

            tokenized_docs = map(lambda doc: simple_preprocess(doc['summary']), generate_examples(file_path, doc_limit=args.doc_limit, byte_offsets=offsets))
            phrase_model = Phrases.load(os.path.join(models_path, "phrase_model.pkl"))

            stopwords = frozenset(set(STOPWORDS) | {"xcite", "xmath", "fig", "eq"})

            for doc in tokenized_docs:
                filtered_doc = filter(lambda word: word not in stopwords, phrase_model[doc])
                dic.add_documents([filtered_doc])
            
            dic.save(os.path.join(models_path, f"counts{id:03}.dict"))
            print(f"Dictionary {id:03} time: {round(time.time() - t, 2)}s")
        
        #------------------------------------------------------------------------------------------

        total_t = time.time()

        #Train phrase model in a separate process
        phrase_process = None
        if args.phrase:
            phrase_process = Process(target=train_phrase_model)
            phrase_process.start()

        #Add docs to elasticsearch
        if not args.no_index:
            create_index(client, index_name)
            indexing()
        
        #Wait for phrase model to finish before training dictionary
        if phrase_process:
            phrase_process.join()

        if args.dict:
            #Train the dictionary
            #The docs are divided by the number of processes
            #Each process takes exactly one batch of lines (docs)
            #(alternatively we could have created batches of lines and allowed processes to take them dynamically)
            workload = divide(args.nprocs, file_batch(file_path, 1))

            with Pool(processes=args.nprocs) as pool:
                pool.starmap(train_dictionary, zip(range(args.nprocs), workload))

            #After each process saves its dictionary, we merge them into one
            print("Merging dictionaries...")
            t = time.time()

            dic = Dictionary.load(os.path.join(models_path, "counts000.dict"))
            os.remove(os.path.join(models_path, "counts000.dict"))

            for dict_name in map(lambda i: os.path.join(models_path, f"counts{i:03}.dict"), range(1, args.nprocs)):
                dic.merge_with(Dictionary.load(dict_name))

            print(f"Merge time: {time.time() - t}")

            #Cleanup
            for file in glob.glob(os.path.join(models_path, "*.dict")):
                os.remove(file)

            #Save final dictionary
            print("Saving final dictionary...")
            dic.save(os.path.join(models_path, "counts.dict"))
            dic.save_as_text(os.path.join(models_path, "counts.txt"))
            
        print(f"Total time: {round(time.time() - total_t)}s")