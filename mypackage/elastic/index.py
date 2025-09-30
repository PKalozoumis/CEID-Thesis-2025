from elasticsearch import Elasticsearch
import json
import sys

#===================================================================================

def create_index(client: Elasticsearch, index_name: str, mapping_path: str = "mapping.json"):
    '''
    Creates an Elasticsearch index

    Arguments
    ---
    client: Elasticsearch
        The Elasticsearch client
    index_name: str
        Name of the new index. Index is deleted if it already exists.
    mapping_path: str
        Path to the mapping file. Defaults to ```mapping.json```
    '''
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
    '''
    Creates an Elasticsearch index

    Arguments
    ---
    client: Elasticsearch
        The Elasticsearch client
    index_name: str
        Name of the new index. Index is deleted if it already exists.
    '''
    if client.indices.exists(index=index_name):
        resp = client.delete_by_query(index=index_name, body={
            "query": {
                "match_all": {}
            }
        })

        #print(resp)
        print(f"Emptied Elasticsearch index {index_name}")

#===================================================================================

