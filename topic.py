from elasticsearch import Elasticsearch
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from elastic import elasticsearch_client
from gensim.models.phrases import Phrases, FrozenPhrases

#===========================================================================================

class CustomCorpus:
    '''
    Receives batches of documents from Elasticsearch
    For each batch, it converts the documents to their vector representation
    '''

    def __init__(self, client: Elasticsearch, index_name: str, dictionary: Dictionary | str, phrase_model: Phrases | str,*, batch_size: int = 10, scroll_time: str="5s"):
        self.client = client
        self.index_name = index_name

        #Load dictionary
        if isinstance(dictionary, Dictionary):
            self.dictionary = dictionary
        elif type(dictionary) is str:
            self.dictionary = Dictionary.load(dictionary)
        else:
            self.dictionary = None

        #Load phrase model
        if isinstance(phrase_model, Phrases) or isinstance(phrase_model, FrozenPhrases):
            self.phrase_model = phrase_model
        elif type(phrase_model) is str:
            self.phrase_model = Phrases.load(phrase_model)
        else:
            self.phrase_model = None
        
        self.batch_size = batch_size
        self.scroll_time = scroll_time

    def __iter__(self):
        res = self.client.search(index=self.index_name, scroll=self.scroll_time, filter_path="_scroll_id,hits.hits", body={
            "size": self.batch_size,
            "_source": ["abstract", "record_number"],
            "query":{"match_all": {}}
        })

        while True:
            if 'error' in res:
                print(res['error']['root_cause'])
                break

            scroll_id = res['_scroll_id']
            docs = res['hits']['hits']

            if docs:
                docs = map(self.dictionary.doc2bow, map(lambda doc: self.phrase_model[simple_preprocess(doc['_source']['abstract'])], docs))

                #Send entire batch before asking for the next one
                for doc in docs:
                    yield doc

                res = self.client.scroll(scroll_id = scroll_id, scroll = self.scroll_time)
            else:
                break

#===========================================================================================

if __name__ == "__main__":
    client = elasticsearch_client("credentials.json", "http_ca.crt")
    index_name = "test-index"
    corpus = CustomCorpus(client, index_name, "counts.dict")

