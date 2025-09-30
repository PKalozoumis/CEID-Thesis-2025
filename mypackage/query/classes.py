from __future__ import annotations
    
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from ..elastic import ElasticDocument, Session

#==============================================================================================

class Query():
    '''
    A class representing an Elasticsearch query.
    '''
    id: str #Query ID
    text: str #The actual query
    source: list[str]
    text_path: str
    vector: np.ndarray

    #------------------------------------------------------------------------------------------

    def __init__(self, id: str, text: str, *, source: list[str] = [], text_path: str | None = None):
        '''
        A class representing an Elasticsearch query.
        
        Arguments
        ---
        id: str
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
        self.source = source
        self.text_path = text_path
        self.vector = None

    #------------------------------------------------------------------------------------------

    def execute(self, sess: Session, size: int = 10) -> list[ElasticDocument]:
        '''
        Execute the query on the specified Elasticsearch session

        Arguments
        ---
        sess: Session
            The Elasticsearch session to run the query on
        size: int
            Number of relevant documents to return. Defaults to ```10```

        Returns
        ---
        docs: list[ElasticDocument]
            The relevant documents to the query
        '''
        search_body = {
            "_source": self.source,
            "query": {
                "multi_match": {
                    "query": self.text,
                    "fields": ["summary^1.5", "article"]
                }
            },
            "size": size
        }

        results = sess.client.search(index=sess.index_name, body=search_body)
        #console.print(results)
        results = results['hits']['hits']

        doc_list = []

        for res in results:
            filter_path = ",".join(["_source." + s for s in self.source])
            elastic_doc = ElasticDocument(sess, int(res['_id']), filter_path=filter_path, text_path=self.text_path)
            
            #Store to cache
            if sess.cache_dir is not None:
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

    #---------------------------------------------------------------------------

    def data(self) -> dict:
        return {
            'id': self.id,
            'text': self.text,
            'source': self.source,
            'text_path': self.text_path
        }
    
    #---------------------------------------------------------------------------

    @classmethod
    def from_data(cls, data: dict) -> Query:
        obj = cls.__new__(cls)
        obj.id = data['id']
        obj.text = data['text']
        obj.source = data['source']
        obj.text_path = data['text_path']

        return obj
