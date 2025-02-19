from elasticsearch import Elasticsearch
import json
import sys
import os
from lxml import etree
from fnmatch import fnmatch
import re
from itertools import chain
from more_itertools import divide
from functools import partial
from multiprocessing import Pool
import helper

index_name = "test-index"
client = helper.elasticsearch_client()

#===================================================================================

def parse_xml(file) -> list[str]:
    f = open(file)
    contents = f.read()

    #Remove empty tags
    res = re.sub(r'<\s*/?\s*?>', '', contents)

    #Split file into documents based on the <RECORD> element
    res = re.split(r'(?=<RECORD>)', res)

    docs = []

    for doc in res[1:]:
        docs.append(etree.fromstring(doc, etree.XMLParser(recover=True)))
    
    f.close()

    return docs

#===================================================================================

def to_json(doc):

    #print(f"Mapping {doc.xpath('//RECORDNUM/text()')[0].strip()}")

    abstract = (doc.xpath("//ABSTRACT/text()") + doc.xpath("//EXTRACT/text()"))

    if not abstract:
        abstract = ""
    else:
        abstract = abstract[0]

    id = doc.xpath("//RECORDNUM/text()")[0].strip()

    yield {"index": {"_id": id}}

    yield {
        "paper_number": doc.xpath("//PAPERNUM/text()")[0],
        "record_number": id,
        "medline_num": doc.xpath("//MEDLINENUM/text()")[0],
        "authors": doc.xpath("//AUTHORS/AUTHOR/text()"),
        "title": doc.xpath("//TITLE/text()")[0].replace("\n", " "),
        "source": doc.xpath("//SOURCE/text()")[0],
        "major_subjects": list(map(lambda x: x.replace("-", " "), doc.xpath("//MAJORSUBJ/TOPIC/text()"))),
        "minor_subjects": list(map(lambda x: x.replace("-", " "), doc.xpath("//MINORSUBJ/TOPIC/text()"))),
        "abstract": abstract.replace("\n", " "),
        "citations": [{
                "author": cite.attrib["author"],
                "publication": cite.attrib["publication"],
                "d1": cite.attrib["d1"],
                "d2": cite.attrib["d2"],
                "d3": cite.attrib["d3"]
            }
        
            for cite in doc.xpath("//CITATIONS/CITE")
                
        ],
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

#===================================================================================

def create_index():
    with open("mapping.json", "r") as f:
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

def empty_index(client: Elasticsearch, index: str):
    resp = client.delete_by_query(index=index, body={
        "query": {
            "match_all": {}
        }
    })

    print(resp)

#===================================================================================

if __name__ == "__main__":
    create_index()
    collection_path = "collection"
    docs = []
    nprocs = 6

    docs = divide(
        nprocs,
        map(
            partial(os.path.join, collection_path),
            filter(
                lambda file: re.match(r"cf\d{2}\.xml", file),
                os.listdir(collection_path)
            )
        )
    )

    def work(docs_chunk):
        docs = chain.from_iterable(map(parse_xml, docs_chunk))
        bulk_data = list(map(json.dumps, chain.from_iterable(map(to_json, docs))))
        bulk_data = "\n".join(bulk_data) + "\n"
        client.bulk(index=index_name, body=bulk_data)

    with Pool(processes=nprocs) as pool:
        results = pool.map(work, docs)