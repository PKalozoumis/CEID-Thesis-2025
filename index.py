from elasticsearch import Elasticsearch
import json
import sys
import os
from fnmatch import fnmatch
import re
from itertools import chain
from more_itertools import divide
from functools import partial
from multiprocessing import Pool
from elastic import elasticsearch_client
from collection_helper import to_json, parse_xml, get_abstract, count_vectorizer
from scipy import sparse
import argparse
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from functools import reduce
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

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

    index_name = "test-index"
    client = elasticsearch_client()

    if args.empty:
        empty_index(client, index_name)
    else:
        create_index(client, index_name)

        collection_path = "collection"
        docs = []

        files = map(
            partial(os.path.join, collection_path),
            filter(
                lambda file: re.match(r"cf\d{2}\.xml", file),
                os.listdir(collection_path)
            )
        )

        docs = list(chain.from_iterable(map(parse_xml, files)))
        tokenized_docs = [simple_preprocess(get_abstract(doc)) for doc in docs]

        #Train and save model
        phrase_model = Phrases(tokenized_docs, 6, 15, connector_words=ENGLISH_CONNECTOR_WORDS)
        phrase_model.save("phrase_model.pkl")

        #For every tokenized document, apply phrase model
        #Then, filter out the remaining stopwords
        #Train dictionary
        #print(list(filter(lambda word: word not in STOPWORDS, phrase_model[tokenized_docs[0]])))

        dic = Dictionary([filter(lambda word: word not in STOPWORDS, phrase_model[doc]) for doc in tokenized_docs])

        print(dic.dfs[dic.token2id["cystic_fibrosis"]])

        bulk_data = list(map(json.dumps, chain.from_iterable(map(to_json, docs))))
        bulk_data = "\n".join(bulk_data) + "\n"
        client.bulk(index=index_name, body=bulk_data)

        dic.save("counts.dict")
        dic.save_as_text("counts.txt")