from elasticsearch import Elasticsearch
import json
import sys
import os
import re
from itertools import chain
from functools import partial
from multiprocessing import Pool, Array
from elastic import elasticsearch_client
from collection_helper import generate_examples, to_bulk_format
import argparse
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import threading
import time
from itertools import islice, tee
from more_itertools import divide

def total_size(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, list):
        size += sum(total_size(item) for item in obj)
    return size / 1024 // 1024

def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

def file_batch(fname: str, batch_size: int):
    byte_offset = 0
    current_lines = 0
    
    yield 0

    with open(fname, "r") as f:
        for line in f:
            byte_offset += len(line.encode('utf-8'))

            if current_lines == batch_size - 1:
                yield byte_offset
                current_lines = 0
            else:
                current_lines += 1
#===================================================================================

def create_index(client: Elasticsearch, index_name: str, mapping_path: str = "mapping.json"):
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1
        },

        "mappings": mapping
    }

    client.indices.delete(index=index_name, ignore_unavailable=True)

    resp = client.indices.create(index=index_name, body=index_settings)

    if resp.get("acknowledged"):
        print("Created index")
    else:
        print("Not acknowledged")
        print(resp["error"]["root_cause"][0]["reason"])
        sys.exit()

#===================================================================================

def empty_index(client: Elasticsearch, index_name: str):
    resp = client.delete_by_query(index=index_name, body={
        "query": {
            "match_all": {}
        }
    })

    print(resp)
    print(f"\nEmpties Elasticsearch index {index_name}")

#===================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Elasticsearch Index Management")
    parser.add_argument("--empty", action="store_true", help="Empty index")
    args = parser.parse_args()

    index_name = "arxiv-index"
    client = elasticsearch_client()

    if args.empty:
        empty_index(client, index_name)
    else:
        create_index(client, index_name)

        collection_path = "collection"
        models_path = "models"

        nprocs = 5
        batch_size = 1800

        #A list of iterables
        #Each iterable has the line offsets the respective process will take
        workload = divide(nprocs, file_batch("collection/test.txt", 1))

        shared_array = Array('i', nprocs)
        shared_array[0] = 1

        def init(arr):
            global locks
            locks = arr

        def work(offsets, id):
            offsets1, offsets2 = tee(offsets, 2)

            tokenized_docs = map(lambda doc: simple_preprocess(doc['article']), generate_examples(os.path.join(collection_path, "test.txt"), offsets1))
            phrase_model = Phrases(tokenized_docs, 6, 15, connector_words=ENGLISH_CONNECTOR_WORDS)

            print(phrase_model.export_phrases())

            bulk = to_bulk_format(generate_examples(os.path.join(collection_path, "test.txt"), offsets2))

            #Each process needs to further divide its lines into batches of batch_size docs
            batches = batched(bulk, 2*batch_size)

            while locks[id] == 0:
                pass            

            #Submit to elasticsearch
            print(id)
            for batch in batches:
                client.bulk(index=index_name, operations=batch)

            #Unlock next process
            if id < nprocs - 1:
                locks[id+1] = 1

        t = time.time()

        with Pool(processes=nprocs, initializer=init, initargs=(shared_array, )) as pool:
            results = pool.starmap(work, zip(workload, list(range(nprocs))))

        print(time.time() - t)