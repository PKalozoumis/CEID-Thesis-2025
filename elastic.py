import metrics
import json
from elasticsearch import Elasticsearch, AuthenticationException
import sys
import os
from collection_helper import Query
from itertools import chain
from typing import Iterable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DocumentList:
    '''
    Lazy document list
    '''
    def __init__(self, client: Elasticsearch, index_name: str, doc_ids: Iterable, filter_path: str ="_source"):
        self.doc_ids = doc_ids
        self.client = client
        self.index_name= index_name
        self.filter_path = filter_path

        self.docs = [None for _ in doc_ids]

    def __getitem__(self, pos: int) -> str:
        assert(type(pos) is int)

        if self.docs[pos] is None:
            self.docs[pos] = self.client.get(index=self.index_name, id=f"{self.doc_ids[pos]:05}", filter_path=self.filter_path)["_source"]

        return self.docs[pos]
    
    def __len__(self) -> int:
        return len(self.doc_ids)
    
    def __iter__(self):
        for i in range(len(self.doc_ids)):
            yield self[i]

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

def query(client: Elasticsearch, index_name: str, query_list: Query | list[Query], filter_path: str = "_source") -> tuple[list[DocumentList], list[list[int]]]:
    '''
    Perform a single query (Query) or multiple queries (list[Query]) and get back results
    
    Returns:
    - A single DocumentList (for one query) or a list of DocumentLists (for list of queries)
    - A single list of document IDs, or a list of lists
    '''
    if type(query_list) is Query:
        query_list = [query_list]

    multiple_query_results = []
    docs = []

    for query in query_list:
        search_body = {
            "match": {"abstract": query.text}
        }

        res = client.search(index=index_name, query=search_body, filter_path=["hits.hits._source.abstract", "hits.hits._id"])
        if len(res) == 0:
            return iter(()), []
        
        res = res['hits']

        id_list = [int(temp['_id']) for temp in res['hits']]
        #docs.append([temp['_source']['abstract'] for temp in res['hits']])
        multiple_query_results.append(id_list) 
        docs.append(DocumentList(client, index_name, id_list, filter_path))
        

    if len(query_list) == 1:
        return docs[0], multiple_query_results[0]
    else:
        return docs, multiple_query_results
    
#===============================================================================================
    
def docs_to_texts(doc_list: DocumentList) -> "map[str]":
    return map(lambda x: x["abstract"], doc_list)

#===============================================================================================

if __name__ == "__main__":
    index_name = "test-index"

    single_query = True

    client = elasticsearch_client()
    queries = parse_queries() #99 queries

    if single_query:
        queries = queries[98]
        docs, single_query_results = query(client, index_name, queries)

        print(single_query_results)
        
        print(f"NDCG: {metrics.ndcg(single_query_results, queries)}")
        print(docs[0])
    else:
        client = elasticsearch_client()
        #queries = parse_queries()[98:99] #99 queries
        queries = parse_queries() #99 queries

        docs, multiple_query_results = query(client, index_name, queries)
        
        print(f"Average NDCG: {metrics.average_ndcg(multiple_query_results, queries)[-1]}")

        print(docs[98][0])