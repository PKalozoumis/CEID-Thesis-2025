import json
from elasticsearch import Elasticsearch, AuthenticationException
import sys
import os
import json

#================================================================================================================

def elasticsearch_client(credentials_path: str = "credentials.json", cert_path: str = "http_ca.crt") -> Elasticsearch:
    '''
    Creates a connection to Elasticsearch

    Arguments
    ---
    credentials_path: str
        The path to the Elasticsearch credentials file. Defaults to ```credentials.json```
    cert_path: str
        The path to the Elasticsearch certificate. Defaults to ```http_ca.crt```

    Returns
    ---
    client: Elasticsearch
        The Elasticsearch client
    '''

    with open(credentials_path, "r") as f:
        credentials = json.load(f)

    client = Elasticsearch(
        "https://localhost:9200",
        ca_certs=cert_path,
        basic_auth=(credentials['elastic_user'], credentials['elastic_password'])
    )\
    .options(ignore_status=400)

    try:
        info = client.info()

    except AuthenticationException:
        print("Wrong password idiot")
        sys.exit()

    return client