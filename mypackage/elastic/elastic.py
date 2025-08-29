import json
from elasticsearch import Elasticsearch, AuthenticationException, NotFoundError
import sys
import os
import json
import time

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

    max_retries = 5
    for attempt in range(0, max_retries + 2):
        try:
            client = Elasticsearch(
                "https://localhost:9200",
                ca_certs=cert_path,
                basic_auth=(credentials['elastic_user'], credentials['elastic_password'])
            )\
            .options(ignore_status=400)

            if not client.ping():
                raise Exception
        except Exception as e:
            if attempt < max_retries:
                print(f"Could not connect to Elasticsearch database. Retrying... ({attempt+1}/{max_retries})")
                time.sleep(10)
            else:
                raise e

    try:
        info = client.info()

    except AuthenticationException:
        print("Wrong password idiot")
        sys.exit()

    return client