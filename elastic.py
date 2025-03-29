import metrics
import json
from elasticsearch import Elasticsearch, AuthenticationException
import sys
import os
from collection_helper import Query
from itertools import chain
from typing import Iterable
from collections import namedtuple
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Session = namedtuple("Session", ["client", "index_name"])

#================================================================================================================

class Document:
    '''
    Class for retrieving and storing a single document from Elasticsearch
    '''

    def __init__(self, session: Session, id: int, *, filter_path: str = "_source", text_path: str | None = None):
        '''
        **session**: An Elasticsearch session\n
        **id**: The numeric ID of the requested document in Elasticsearch\n
        **filter_path**: Path to the field(s) to keep from the response body. If path leads to a single field, then its contents will be returned instead 
        **text_path**: Name of the field (after filter_path is applied, if specified) where the document's body is located\n
        '''
        self.doc = None
        self.client = session.client
        self.index_name = session.index_name
        self.id = id
        self.filter_path = filter_path

        if type(self.filter_path) is str and type(text_path) is str:

            matched = False

            #Check if the text field is contaned in any of the filter paths
            paths = self.filter_path.split(",")
            for path in paths:
                if not (text_path == path or text_path.startswith(path + ".")):
                    continue
                matched = True

                #If multiple filter paths, then the full text field path stays as is
                if len(paths) > 1:
                    self.text_field = text_path
                else:                
                    pattern = re.compile(f"{path}\.(\w+)(\..+)?")
                    self.text_field = re.sub(pattern, r"\1\2", text_path)
                    break

            if not matched:
                raise ValueError(f"Text field path {text_path} not contained in filter path {self.filter_path}")

    def get(self):
        if self.doc is None:
            self.doc = self.client.get(index=self.index_name, id=f"{self.id}", filter_path=self.filter_path)

            if self.filter_path and len(self.filter_path.split(",")) == 1:
                for key in self.filter_path.split("."):
                    self.doc = self.doc[key]

        return self.doc
    
    def __str__(self):
        return json.dumps(self.get())
    
    def text(self):
        if self.text_field is None:
            return
        
        temp = self.doc
            
        if temp is None:
            temp = self.get()

        #If the document is a dictionary, then we can (potentially) traverse the text_field path more, until we find the final field
        #Else we just return the single field
        if type(temp) is dict:
            for key in self.text_field.split("."):
                temp = temp[key]

        return temp

        
#================================================================================================================

class DocumentList:
    '''
    Lazy document list
    '''
    def __init__(self, session: Session, doc_ids: Iterable, filter_path: str ="_source"):
        self.doc_ids = doc_ids
        self.client = session.client
        self.index_name= session.index_name
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

class ScrollingCorpus:
    '''
    Receives batches of documents from Elasticsearch
    '''

    def __init__(self,   
            session: Session,
            *,
            batch_size: int = 10,
            scroll_time: str="5s",
            doc_field: str,
            fields_to_keep: list[str] = []
        ):
        self.client = session.client
        self.index_name = session.index_name
        self.batch_size = batch_size
        self.scroll_time = scroll_time
        self.fields_to_keep = fields_to_keep
        self.doc_field = doc_field

        if self.doc_field:
            self.fields_to_keep.append(self.doc_field)

    #--------------------------------------------------------------------------------------

    def __iter__(self):
        res = self.client.search(index=self.index_name, scroll=self.scroll_time, filter_path="_scroll_id,hits.hits", body={
            "size": self.batch_size,
            "_source": self.fields_to_keep,
            "query":{"match_all": {}}
        })

        while True:
            if 'error' in res:
                print(res['error']['root_cause'])
                break

            scroll_id = res['_scroll_id']
            docs = res['hits']['hits']

            if docs:
                #Send entire batch before asking for the next one
                docs = map(lambda doc: doc['_source'][self.doc_field], docs)
                yield from docs
                res = self.client.scroll(scroll_id = scroll_id, scroll = self.scroll_time)
            else:
                break

#================================================================================================================

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

#================================================================================================================

def elastic_session(index_name: str, credentials_path: str = "credentials.json", cert_path: str = "http_ca.crt") -> Session:
    return Session(elasticsearch_client(credentials_path, cert_path), index_name)

#===============================================================================================

def query(session: Session, query_list: Query | list[Query], filter_path: str = "_source") -> tuple[list[DocumentList], list[list[int]]]:
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

        res = session.client.search(index=session.index_name, query=search_body, filter_path=["hits.hits._source.abstract", "hits.hits._id"])
        if len(res) == 0:
            return iter(()), []
        
        res = res['hits']

        id_list = [int(temp['_id']) for temp in res['hits']]
        #docs.append([temp['_source']['abstract'] for temp in res['hits']])
        multiple_query_results.append(id_list) 
        docs.append(DocumentList(session.client, session.index_name, id_list, filter_path))
        

    if len(query_list) == 1:
        return docs[0], multiple_query_results[0]
    else:
        return docs, multiple_query_results
    
#===============================================================================================
    
def docs_to_texts(doc_list: DocumentList) -> "map[str]":
    return map(lambda x: x["abstract"], doc_list)

#===============================================================================================

if __name__ == "__main__":
    
    session = elastic_session("arxiv-index")
    docs = [Document(session, i, text_path="_source.abstract") for i in range(10)]

    print(docs)

    '''
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
        '''