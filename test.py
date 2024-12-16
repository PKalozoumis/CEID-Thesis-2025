from elasticsearch import Elasticsearch
from elasticsearch import AuthenticationException
import json
import sys
import os
from bs4 import BeautifulSoup
from lxml import etree

collection_path = "collection"

with open("credentials.json", "r") as f:
    credentials = json.load(f)

with open("mapping.json", "r") as f:
    mapping = json.load(f)

doc = etree.parse(os.path.join(collection_path, "test.xml")).getroot()

json_doc = {
    "paper_number": doc.xpath("//PAPERNUM/text()")[0],
    "record_number": doc.xpath("//RECORDNUM/text()")[0],
    "medline_num": doc.xpath("//MEDLINENUM/text()")[0],
    "authors": doc.xpath("//AUTHORS/AUTHOR/text()"),
    "title": doc.xpath("//TITLE/text()")[0],
    "source": doc.xpath("//SOURCE/text()")[0],
    "major_subjects": doc.xpath("//MAJORSUBJ/TOPIC/text()"),
    "minor_subjects": doc.xpath("//MINORSUBJ/TOPIC/text()"),
    "abstract": doc.xpath("//ABSTRACT/text()")[0],
    "references": [{
            "author": cite.attrib["author"],
            "publication": cite.attrib["publication"],
            "d1": cite.attrib["d1"],
            "d2": cite.attrib["d2"],
            "d3": cite.attrib["d3"]
        }

        for cite in doc.xpath("//REFERENCES/CITE")
    ]
}

#print(json.dumps(json_doc, indent=2))

client = Elasticsearch(
    "https://localhost:9200",
    ca_certs="http_ca.crt",
    basic_auth=("elastic", credentials["elastic_password"])
)

try:
    info = client.info()
    #print(info)

except AuthenticationException:
    print("Wrong password idiot")
    sys.exit()


index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1
    },

    "mappings": mapping
}

client = client.options(ignore_status=400)
resp = client.indices.create(index="dista-index", body=mapping)

if resp.get("acknowledged"):
    print("Dista acknowledged")
else:
    print("Dista not acknowledged")

resp = client.index(index="dista-index", id=1, document=json_doc)
print(resp["result"])

resp = client.get(index="dista-index", id=1)

#Query
#=================================================================================
search_body = {
    "query": {
        "match_all": {}
    }
}

resp = client.search(index="dista-index", body=search_body)

for hit in resp['hits']['hits']:
    print(hit['_source'])