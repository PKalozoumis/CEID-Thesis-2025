from elasticsearch import Elasticsearch, AuthenticationException
from dataclasses import dataclass, field
from ..helper import overrides
from typing import Iterable, Any, NamedTuple
import os
import json
from .elastic import elasticsearch_client
from elastic_transport import ObjectApiResponse
import warnings
from rich.console import Console

console = Console()

#================================================================================================================

class Score(NamedTuple):
    s1: int
    s2: int
    s3: int
    s4: int

#================================================================================================================

class Session():

    #Actually even better: replace cache_dir and use with 'source'
    
    client: Elasticsearch
    index_name: str
    cache_dir: str
    use: str

    def __init__(
            self,
            index_name: str,
            *
            ,
            client: Elasticsearch = None,
            cache_dir: str = None,
            base_path: str = None,
            credentials_path: str = "credentials.json",
            cert_path: str = "http_ca.crt",
            use: str = "client"
        ):
        '''
        Represents an Elasticsearch session for querying and retrieving documents from a specific index.
        Groups information about the client, the index and the authentication methods

        Arguments
        ---
        index_name: str
            The index to connect to

        client: Elasticsearch, optional
            The Elasticsearch client. If not set, it's automatically generated from ```credentials_path``` and ```cert_path```.
            Either ```client/credentials``` or ```cache_dir``` needs to be set. Using both gives priority to the cache
            for retrieval

        cache_dir: str, optional
            An alternative retrieval method to the Elasticsearch client. A cache directory to retrieve and store documents.
            When using the cache instead of the client, we can skip providing the credentials and the certificate.
            Either ```cache_dir``` or ```client/credentials``` needs to be set. Using both gives priority to the cache
            for retrieval

        credentials_path: str
            Path to a JSON file containing the Elasticsearch username (```elastic_user```), password (```elastic_password```)
            and certificate fingerprint (```cert_fingerprint```). If not set, defaults to ```credentials.json```

        cert_path: str
            Path to the Elasticsearch certificate. If not set, defaults to ```http_ca.crt```

        base_path: str, optional
            By default, the files ```credentials.json``` and ```http_ca.crt``` should be in the directory we run the script from.
            We can change the directory where these files are searched for by setting ```base_path```.
            This is useful if we want to maintain the default names, but the files are stored somewhere else

        use: str
            Which method to use to retrieve documents.
            Possible values are ```client```, ```cache``` and ```both```. Defaults to ```client```
        '''
        self.client = client
        self.cache_dir = cache_dir
        self.use = use
        self.index_name = index_name

        if use == 'cache' or use == 'both':
            if self.cache_dir and not os.path.isdir(self.cache_dir):
                raise Exception(f"The cache directory {self.cache_dir} does not exist")

        if use == 'client' or use == 'both':
            if not self.client:
                if base_path:
                    credentials_path = os.path.join(base_path, credentials_path)
                    cert_path = os.path.join(base_path, cert_path)
                
                self.client = elasticsearch_client(credentials_path, cert_path)

    #----------------------------------------------------------------------------------------------

    def cache_store(self, raw_elasticsearch_doc: ObjectApiResponse, id: str):
        if self.cache_dir:
            fname = os.path.join(self.cache_dir, f"{self.index_name.replace('-', '_')}_{id:04}.json")

            with open(fname, "w") as f:
                json.dump(dict(raw_elasticsearch_doc), f, ensure_ascii=False)
        else:
            warnings.warn(f"Called Session.cache_store, but the cache_dir was not set")

    #----------------------------------------------------------------------------------------------

    def cache_load(self, id: str) -> dict | None:
        if self.use in ["cache", "both"] and self.cache_dir:
            fname = os.path.join(self.cache_dir, f"{self.index_name.replace('-', '_')}_{id:04}.json")
            if os.path.isfile(fname):
                with open(fname, "r") as f:
                    return json.load(f)
        else:
            return None

#================================================================================================================

@dataclass
class Document():
    '''
    A class representing a document. Does not necessarily have to be an Elasticsearch document

    Args:
        doc (Any): The document's contents. Typically ```str``` or ```dict```, but it can be any other type
        id (int, optional): A numeric ID for the document. For Elasticsearch documents, corresponds to the ID in the index
        text_path (str, optional): Path to the document's main body where the text is located. Only applicable when document is of type ```dict```
    '''
    doc: Any
    id: str = field(default=None)
    text_path: str = field(default=None)

    def __post_init__(self):
        self.id = str(self.id)

    @classmethod
    def from_json(cls, path: str, id: str = None, text_path: str = None) -> 'Document':
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
                raise ValueError("Called Document.text on a dictionary document, but text_path was None")
        
            for key in self.text_path.split("."):
                if key not in temp:
                    raise ValueError(f"Key '{key}' in text_path '{self.text_path}' does not exist")

                temp = temp[key]
        elif self.text_path is not None:
            warnings.warn(f"text_path '{self.text_path}' ignored, as the document is not a dictionary. Both text and get() will return the same result")

        return temp
    
    #--------------------------------------------------------------------------------

    def __str__(self):
        return json.dumps(self.get())
    
    def __repr__(self):
        return f"Document(id={self.id})"
    
    #--------------------------------------------------------------------------------

    def __eq__(self, other: 'Document') -> bool:
        return self.id == other.id
    
    def __hash__(self):
        return hash(id)
#================================================================================================================

class ElasticDocument(Document):
    '''
    A class representing a documement in an Elasticsearch index. Can retrieve and store a single document.
    '''
    doc: str|dict
    id: str
    text_path: str|None
    session: Session
    filter_path: str

    def __init__(self, session: Session, id: str, *, filter_path: str = "_source", text_path: str | None = None):
        '''
        A class representing a documement in an Elasticsearch index. Can retrieve and store a single document.

        Arguments
        ---
        session: Session
            An Elasticsearch session

        id: int
            The numeric ID of the requested document in Elasticsearch

        filter_path: str
            Comma-separated paths to the field(s) to keep from the response body. If path leads to a single field, then its contents will be returned instead. Defaults to ```_source```
        
        text_path: str, optional
            Name of the field (after filter_path is applied, if specified) where the document's body is located
        '''
        super().__init__(None, id, text_path)

        self.session = session
        self.filter_path = filter_path

    #--------------------------------------------------------------------------------

    @overrides(Document)
    def get(self):
        if self.doc is None:
            #print(f"Fetching Document(ID={self.id})")

            #Try to load from cache first
            self.doc = self.session.cache_load(self.id)

            #If loading from cache failed
            if self.doc is None:
                if self.session.client is None:
                    if self.session.use == "cache":
                        raise Exception("Session is in 'cache' mode, but the document was not found in cache. Consider setting mode to 'client' or 'both'")
                self.doc = self.session.client.get(index=self.session.index_name, id=f"{self.id}", filter_path=self.filter_path)
                self.session.cache_store(self.doc, self.id)

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

#==============================================================================================

class Query():
    '''
    A class representing an Elasticsearch query.
    '''
    id: int #Query ID
    text: str #The actual query
    match_field: str 
    source: list[str]
    text_path: str

    #------------------------------------------------------------------------------------------

    def __init__(self, id: int, text: str, *, match_field: str = "article", source: list[str] = [], text_path: str | None = None, cache_dir: str | None = None):
        '''
        A class representing an Elasticsearch query.
        
        Arguments
        ---
        id: int
            The query ID

        text: str
            The query

        source: list[str], optional
            List of the fields to return from the document. Corresponds to the ```_source``` argument in Elasticsearch

        text_path: str | None, optional
            Name of the field inside the document (inside ```_source```) that is the main text.
            This is forwarded to the generated documents' ```text_path``` argument

        cache_dir: str | None, optional
            Path to cache the returned documents in
        '''
        self.id = id
        self.text = text
        self.match_field = match_field
        self.source = source
        self.text_path = text_path
        self.cache_dir = cache_dir

    #------------------------------------------------------------------------------------------

    def execute(self, sess: Session) -> list[ElasticDocument]:

        '''
        search_body = {
            "_source": self.source,
            "query":
            {
                "match": {self.match_field: self.text}
            }
        }
        '''
        search_body = {
            "_source": self.source,
            "query": {
                "multi_match": {
                    "query": self.text,
                    "fields": ["summary^1.5", "article"]
                }
            }
        }

        results = sess.client.search(index=sess.index_name, body=search_body)
        #console.print(results)
        results = results['hits']['hits']

        doc_list = []

        for res in results:
            filter_path = ",".join(["_source." + s for s in self.source])
            elastic_doc = ElasticDocument(sess, res['_id'], filter_path=filter_path, text_path=self.text_path, cache_dir=self.cache_dir)
            
            #Store to cache
            elastic_doc.cache(res)
            temp_doc = res

            if len(res['_source']) > 0:
                if filter_path and len(filter_path.split(",")) == 1:
                    for key in filter_path.split("."):
                        temp_doc = temp_doc[key]
                else:
                    temp_doc = res['_source']
            else:
                temp_doc = None
            
            elastic_doc.doc = temp_doc
            doc_list.append(elastic_doc)

        return doc_list

#==============================================================================================

class EvaluableQuery(Query):
    '''
    A class representing an Elasticsearch query. Also contains it's relevant docs for evaluation purposes
    '''

    id: int #Query ID
    text: str #The actual query
    relevant_docs: list[int] #List with the relevant doc IDs

    def __init__(self, id: int, text: str, *, match_field: str = "article", source: list[str] = [], text_path: str | None = None, relevant_docs: list[int]):
        '''
        A class representing an Elasticsearch query. Also contains it's relevant docs for evaluation purposes
        
        Arguments
        ---
        id: int
            The query ID

        text: str
            The query
        '''
        super().__init__(id, text, match_field=match_field, source=source, text_path=text_path)
        self.relevant_docs = relevant_docs

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