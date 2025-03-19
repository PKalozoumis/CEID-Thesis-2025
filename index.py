from elasticsearch import Elasticsearch
import json
import sys
import os
import re
from itertools import chain
from functools import partial
from multiprocessing import Process
from elastic import elasticsearch_client
from collection_helper import generate_examples, to_bulk_format
import argparse
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import threading
import time
from itertools import islice, tee, chain
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

        batch_size = 1500

        total_t = time.time()

        def phrase_model():
            t = time.time()
            tokenized_docs = map(lambda doc: simple_preprocess(doc['article']), generate_examples(os.path.join(collection_path, "test.txt")))
            phrase_model = Phrases(tokenized_docs, 6, 15, connector_words=ENGLISH_CONNECTOR_WORDS)
            print(f"Phrase time: {round(time.time() - t, 2)}s")

            return phrase_model

        def indexing():
            t = time.time()
            bulk = to_bulk_format(generate_examples(os.path.join(collection_path, "test.txt")))
            batches = batched(bulk, 2*batch_size)
            #Add to elasticsearch
            for batch in batches:
                client.bulk(index=index_name, operations=batch)
            print(f"Elastic time: {round(time.time() - t, 2)}s")

        p = Process(target=phrase_model)
        #p = threading.Thread(target=phrase_model)
        p.start()

        indexing()
        
        p.join()

        print(f"Total time: {round(time.time() - total_t)}s")