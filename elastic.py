import metrics
import json
from elasticsearch import Elasticsearch, AuthenticationException
import sys
import os
from collection_helper import Query
from itertools import chain
from typing import Iterable, NamedTuple, Any
import re
from dataclasses import dataclass, field
from helper import overrides

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Session(NamedTuple):
    client: Elasticsearch
    index_name: str

#================================================================================================================

@dataclass
class Document:
    '''
    A class representing a document
    '''
    doc: Any
    id: int
    text_path: str = field(default=None)

    #--------------------------------------------------------------------------------

    def get(self) -> Any:
        return self.doc
    
    #--------------------------------------------------------------------------------
    
    def text(self) -> str:
        temp = self.doc

        if type(temp) is dict:
            if self.text_path is None:
                raise ValueError("Called Document.text() on a dictionary document, but text_path was None")
        
            for key in self.text_path.split("."):
                if key not in temp:
                    raise ValueError(f"Key '{key}' in text_path '{self.text_path}' does not exist")

                temp = temp[key]

        return temp
    
    #--------------------------------------------------------------------------------

    def __str__(self):
        return json.dumps(self.get())
    
#================================================================================================================

class ElasticDocument(Document):
    '''
    A class representing a documement in an Elasticsearch index. Can retrieve and store a single document.
    '''

    def __init__(self, session: Session, id: int, *, filter_path: str = "_source", text_path: str | None = None):
        '''
        **session**: An Elasticsearch session\n
        **id**: The numeric ID of the requested document in Elasticsearch\n
        **filter_path**: Path to the field(s) to keep from the response body. If path leads to a single field, then its contents will be returned instead 
        **text_path**: Name of the field (after filter_path is applied, if specified) where the document's body is located\n
        **doc**: Preloaded document content, if we have already retrieved it somehow. Skips the extra request
        '''
        super().__init__(None, id, text_path)
        self.session = session
        self.filter_path = filter_path

    #--------------------------------------------------------------------------------

    @overrides(Document)
    def get(self):
        if self.doc is None:
            print(f"Fetching Document(ID={self.id})")
            self.doc = self.session.client.get(index=self.session.index_name, id=f"{self.id}", filter_path=self.filter_path)

            #If the filter path points to a single field, return the value inside that field
            if self.filter_path and len(self.filter_path.split(",")) == 1:
                for key in self.filter_path.split("."):
                    self.doc = self.doc[key]

                return self.doc

        self.doc = dict(self.doc)
        return self.doc
    
    #--------------------------------------------------------------------------------
    
    @overrides(Document)
    def text(self):
        self.get()
    
        return super().text()
    
    #--------------------------------------------------------------------------------
    
    def __repr__(self):
        return f"ElasticDocument(id={self.id})"
        
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
    Receives batches of documents from Elasticsearch\n
    Used to retrieve the actual documents only. Cannot return metadata
    '''

    def __init__(self,   
            session: Session,
            *,
            batch_size: int = 10,
            scroll_time: str="5s",
            doc_field: str,
            fields_to_keep: list[str] = []
        ):
        self.session = session
        self.batch_size = batch_size
        self.scroll_time = scroll_time
        self.fields_to_keep = fields_to_keep
        self.doc_field = doc_field

        if self.doc_field:
            self.fields_to_keep.append(self.doc_field)

    #--------------------------------------------------------------------------------------

    def __iter__(self):
        res = self.session.client.search(index=self.session.index_name, scroll=self.scroll_time, filter_path="_scroll_id,hits.hits", body={
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
                for doc in docs:
                    doc_obj = Document(doc['_source'][self.doc_field], doc['_id'])
                    yield doc_obj

                res = self.session.client.scroll(scroll_id = scroll_id, scroll = self.scroll_time)
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

if __name__ == "__main__":

    from rich.panel import Panel
    from rich.console import Console

    console = Console()
    
    session = elastic_session("arxiv-index")
    docs = [
            Panel(
                ElasticDocument(session, i, filter_path="_source.article_id,_source.summary", text_path="_source.summary").text(),
                title="Text",
                title_align="left",
                border_style="cyan"
            )
            for i in range(1)
        ]

    console.print(docs[0])