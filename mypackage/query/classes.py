import numpy as np
from sentence_transformers import SentenceTransformer
import os
from typing import NamedTuple

from ..elastic import ElasticDocument, Session

#================================================================================================================

class Score(NamedTuple):
    s1: int
    s2: int
    s3: int
    s4: int

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
    vector: np.ndarray

    #------------------------------------------------------------------------------------------

    def __init__(self, id: int, text: str, *, match_field: str = "article", source: list[str] = [], text_path: str | None = None):
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
        '''
        self.id = id
        self.text = text
        self.match_field = match_field
        self.source = source
        self.text_path = text_path
        self.vector = None

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
            elastic_doc = ElasticDocument(sess, int(res['_id']), filter_path=filter_path, text_path=self.text_path)
            
            #Store to cache
            sess.cache_store(res, elastic_doc.id)
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
    
    def encode(self, sentence_transformer: SentenceTransformer):
        self.vector = sentence_transformer.encode(self.text)

    #FOR DEBUGGING ONLY. The final version is seen above
    def load_vector(self, sentence_transformer: SentenceTransformer, path: str = "query.npy"):
        if self.vector is not None:
            return

        #if not os.path.exists(path):
        self.encode(sentence_transformer)
        np.save(path, self.vector)
        #else:
            #print("Loading query from disk...")
            #self.vector = np.load(path)


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

#================================================================================================================