from elasticsearch import Elasticsearch
import json
import sys
import os
import re
from itertools import chain
from functools import partial
from multiprocessing import Pool
from elastic import elasticsearch_client
from collection_helper import generate_examples, to_bulk_format
import argparse
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import threading
import time
from itertools import islice

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
        offsets = file_batch("collection/test.txt", 1000)

        with open("collection/test.txt", "r") as f:
            for offset in offsets:
                f.seek(offset)
                print(f.readline()[:100])
        '''
        create_index(client, index_name)

        collection_path = "collection"
        models_path = "models"

        bulk = to_bulk_format(generate_examples(os.path.join(collection_path, "test.txt")))

        batch_size = 1500
        batches = map(lambda batch: "\n".join(batch), batched(bulk, 2*batch_size))

        for batch in batches:
            #print(batch)
            client.bulk(index=index_name, body=batch+"\n")'
        '''