from elasticsearch import Elasticsearch
from elasticsearch import AuthenticationException
import json

with open("credentials.json", "r") as f:
    credentials = json.load(f)

print(credentials)

client = Elasticsearch(
    "https://localhost:9200",
    ca_certs="http_ca.crt",
    basic_auth=("elastic", credentials["elastic_password"])
)

try:
    info = client.info()
    print(info)

except AuthenticationException:
    print("Wrong password idiot")


index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1
    },

    "mappings": {
        "properties": {
            "paper_number": {"type": "string"},
            
        }
    }
}

