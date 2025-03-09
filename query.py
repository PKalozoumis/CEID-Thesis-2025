import metrics
from transformers import pipeline
import json
from elasticsearch import Elasticsearch, AuthenticationException
import sys
import os
from lxml import etree
import re
from helper import Score, Query

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

index_name = "test-index"

#==============================================================================================

def elasticsearch_client(credentials_path: str = "credentials.json", cert_path: str = "http_ca.crt") -> Elasticsearch:

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

#===============================================================================================

def parse_queries(collection_path: str = "collection") -> list[Query]:

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

#===============================================================================================

def query(client: Elasticsearch, query_list: list[Query]) -> tuple[list[list[str]], list[list[int]]]:
    multiple_query_results = []
    docs = []

    for query in query_list:
        search_body = {
            "match": {"abstract": query.text}
        }

        res = client.search(index=index_name, query=search_body, filter_path=["hits.hits._source.abstract", "hits.hits._id"])['hits']
        docs.append([temp['_source']['abstract'] for temp in res['hits']])
        multiple_query_results.append([int(temp['_id']) for temp in res['hits']])

    return docs, multiple_query_results

#===============================================================================================

if __name__ == "__main__":

    client = elasticsearch_client()
    queries = parse_queries()[98:99] #99 queries

    docs, multiple_query_results = query(client, queries)
    
    print(f"Average NDCG: {metrics.average_ndcg(multiple_query_results, queries)[-1]}")

    hyperdoc = ". ".join(docs[0])
    #print(hyperdoc)

    '''
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cpu")

    summary  = ""

    #1024 tokens
    for doc in docs[0]:
        print(f"Original\n{'-'*40}\n{doc}\n")
        temp = summarizer(doc, max_length=len(doc.split(' '))*1.2, min_length=30, do_sample=False)[0]['summary_text']
        print(f"Summary\n{'-'*40}\n{temp}\n")
        print("="*80 + "\n")
        summary += temp + ". "

    print("FINAL RESULT\n\n")

    print(summary)

    print("\n\n")

    print(summarizer(summary, max_length=len(summary.split(' '))*1.8, min_length=30, do_sample=False)[0]['summary_text'])'
    '''