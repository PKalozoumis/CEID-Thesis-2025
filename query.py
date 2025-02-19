from elasticsearch import Elasticsearch, AuthenticationException
import sys
import json
import re
import os
from lxml import etree

from metrics import Score, Query

collection_path = "collection"
index_name = "test-index"

#===============================================================================================

def parse_queries():
    f = open(os.path.join(collection_path, "cfquery.xml"))
    res = re.split(r'(?=<QUERY>)', f.read())
    queries = map(lambda x: etree.fromstring(x, etree.XMLParser(recover=True)), res[1:])

    final_queries = []

    for query in queries:
        items = query.xpath("//Records/Item")

        docs = map(lambda item: int(item.text))

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

#===============================================================================================

if __name__ == "__main__":
    with open("credentials.json", "r") as f:
        credentials = json.load(f)

    with open("mapping.json", "r") as f:
        mapping = json.load(f)

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

    #Query
    #=================================================================================
    queries = parse_queries()
    query = {
        "match": {"abstract": queries[70].text}
    }

    res = client.search(index=index_name, query=query, filter_path=["hits.hits._id"])["hits"]["hits"]
    res = [int(id) for temp in res for id in temp.values()]

    print(res)