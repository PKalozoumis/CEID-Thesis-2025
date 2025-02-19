import json
from elasticsearch import Elasticsearch, AuthenticationException
import sys
from collections import namedtuple
import os
from lxml import etree
import re

Score = namedtuple("Score", ["s1", "s2", "s3", "s4"])
Query = namedtuple("Query", ["id", "text", "num_results", "docs", "scores"])

collection_path = "collection"
index_name = "test-index"

#==============================================================================================

def parse_queries():
    f = open(os.path.join(collection_path, "cfquery.xml"))
    res = re.split(r'(?=<QUERY>)', f.read())
    queries = map(lambda x: etree.fromstring(x, etree.XMLParser(recover=True)), res[1:])

    final_queries = []

    for query in queries:
        items = query.xpath("//Records/Item")

        docs = map(lambda item: int(item.text), items)

        scores = map(lambda item: Score(
            int(item.attrib["score"][0]),
            int(item.attrib["score"][1]),
            int(item.attrib["score"][2]),
            int(item.attrib["score"][3])
        ), items)

        final_queries.append(Query(
            int(query.xpath("//QueryNumber/text()")[0]),
            re.sub(r"\s{2,}", " ", query.xpath("//QueryText/text()")[0].strip()),
            int(query.xpath("//Results/text()")[0]),
            list(docs),
            list(scores)
        ))

    f.close()
    return final_queries

#==============================================================================================

def elasticsearch_client():

    with open("credentials.json", "r") as f:
        credentials = json.load(f)

    client = Elasticsearch(
        "https://localhost:9200",
        ca_certs="http_ca.crt",
        basic_auth=(credentials['elastic_user'], credentials['elastic_password'])
    )\
    .options(ignore_status=400)

    try:
        info = client.info()

    except AuthenticationException:
        print("Wrong password idiot")
        sys.exit()

    return client