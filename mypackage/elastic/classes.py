from elasticsearch import Elasticsearch, AuthenticationException
from dataclasses import dataclass, field
from ..helper import overrides
from typing import Iterable, Any, NamedTuple
import os
import json
from .elastic import elasticsearch_client
from elastic_transport import ObjectApiResponse

#================================================================================================================

class Score(NamedTuple):
    s1: int
    s2: int
    s3: int
    s4: int

#================================================================================================================

class Query(NamedTuple):
    id: int #Query ID
    text: str #The actual query
    num_results: int #Number of retrieved documents
    relevant_doc_ids: list[int]
    scores: list[Score] #Each position contains the relevance Score (used in DCG) for the respective relevant document in relevant_doc_ids

#================================================================================================================

class Session():

    def __init__(self, index_name: str, *, client: Elasticsearch = None, credentials_path: str = "credentials.json", cert_path: str = "http_ca.crt"):
        
        if client:
            self.client = client
        else:
            self.client = elasticsearch_client(credentials_path, cert_path)

        self.index_name = index_name

#================================================================================================================

@dataclass
class Document:
    '''
    A class representing a document. Does not necessarily have to be an Elasticsearch document

    Args:
        doc (Any): The document's contents. Typically ```str``` or ```dict```, but it can be any other type
        id (int, optional): A numeric ID for the document. For Elasticsearch documents, corresponds to the ID in the index
        text_path (str, optional): Path to the document's main body where the text is located. Only applicable when document is of type ```dict```
    '''
    doc: Any
    id: int = field(default=None)
    text_path: str = field(default=None)

    @classmethod
    def from_json(cls, path: str, id: int = None, text_path: str = None) -> 'Document':
        '''
        Create a Document from a JSON file

        Args:
            path (str): Path to the file
            id (int, optional): A numeric ID for the document
            text_path (str, optional): Path to the document's main body where the text is located
        '''
        doc = None
        
        if not os.path.isfile(path):
            raise FileNotFoundError("Path does not lead to existing file")
        
        with open(path, "r") as f:
            doc = json.load(f)

        return cls(doc, id, text_path)

    #--------------------------------------------------------------------------------

    def get(self) -> Any:
        return self.doc
    
    #--------------------------------------------------------------------------------
    
    @property
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
        A class representing a documement in an Elasticsearch index. Can retrieve and store a single document.

        Args:
            session (Session): An Elasticsearch session
            id (int): The numeric ID of the requested document in Elasticsearch
            filter_path (str, optional): Path to the field(s) to keep from the response body. If path leads to a single field, then its contents will be returned instead. Defaults to ```_source```
            text_path (str, optional): Name of the field (after filter_path is applied, if specified) where the document's body is located
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

                #Here, we return the single value
                #This is no longer a dictionary-like object
                return self.doc
            else: #The document is dictionary-like. Specifically, it is an ObjectApiResponse
                assert isinstance(self.doc, ObjectApiResponse)
                self.doc = dict(self.doc)
        else:
            return self.doc
    
    #--------------------------------------------------------------------------------
    
    @property
    @overrides(Document)
    def text(self):
        self.get()
    
        return super().text
    
    #--------------------------------------------------------------------------------
    
    def __repr__(self):
        return f"ElasticDocument(id={self.id})"
        
#================================================================================================================

class DocumentList:
    '''
    Lazy document list
    '''
    def __init__(self, session: Session, doc_ids: Iterable, filter_path: str ="_source", text_path: str | None = None):
        self.doc_ids = doc_ids
        self.client = session.client
        self.index_name= session.index_name
        self.filter_path = filter_path
        self.text_path = text_path

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
    Receives batches of documents from Elasticsearch.
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


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def query(session: Session, query_list: Query | list[Query], filter_path: str = "_source") -> tuple[list[DocumentList], list[list[int]]]:
    '''
    Perform a single query (Query) or multiple queries (list[Query]) and get back results
    
    Arguments
    ---
    session: Session
        The Elasticsearch session

    query_list: Query | list[Query]


    Returns
    ---
    docs: DocumentList | list[DocumentList]:
        A single DocumentList (for one query) or a list of DocumentLists (for list of queries)

    doc_ids: int | list[int]:
        A single list of document IDs, or a list of lists
    '''

    #We treat a single query object as a list with only one query
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
    
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX