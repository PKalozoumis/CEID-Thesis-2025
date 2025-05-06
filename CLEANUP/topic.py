from elasticsearch import Elasticsearch
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from mypackage.elastic.elastic import elasticsearch_client, query, DocumentList, docs_to_texts
from gensim.models.phrases import Phrases, FrozenPhrases
from gensim.models.ldamodel import LdaModel
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

#===========================================================================================

class ScrollingCorpusVectorized:
    '''
    Receives batches of documents from Elasticsearch
    For each batch, it converts the documents to their vector representation
    '''

    def __init__(self,
                 
                 client: Elasticsearch,
                 index_name: str,
                 dictionary: Dictionary | str,
                 phrase_model: Phrases | str,
                 *,
                 batch_size: int = 10,
                 scroll_time: str="5s",
                 doc_field: str,
                 fields_to_keep: list[str]
        ):
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
        self.fields_to_keep = fields_to_keep
        self.doc_field = doc_field

        if self.doc_field:
            self.fields_to_keep.append(self.doc_field)
            print(self.fields_to_keep)

    #================================================================================================================

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
                docs = map(self.dictionary.doc2bow, map(lambda doc: self.phrase_model[simple_preprocess(doc['_source'][self.doc_field])], docs))

                #Send entire batch before asking for the next one
                for doc in docs:
                    yield doc

                res = self.client.scroll(scroll_id = scroll_id, scroll = self.scroll_time)
            else:
                break

    #================================================================================================================

    def doc_to_vec(self, doc: str):
        '''
        Converts any document to a vector
        '''
        return self.dictionary.doc2bow(self.phrase_model[simple_preprocess(doc)])
    
#================================================================================================================
    
def train_lda(corpus: ScrollingCorpusVectorized, num_topics: int) -> LdaModel:
    return LdaModel(corpus, id2word=corpus.dictionary, num_topics=num_topics, alpha="auto", eta="auto")

#================================================================================================================
    
def docs_to_topics(corpus: ScrollingCorpusVectorized, docs_list: list[str] | DocumentList, num_topics: int = 6, pretrained_lda: LdaModel = None) -> tuple[np.ndarray, list]:
    '''
    Takes a list of documents and returns their topic vectors

    - topic_matrix
    - topic_words
    '''
    def topic_to_words(topic: tuple) -> list:
        find = re.findall(r"\"(\w+)\"", topic)
        return find

    if pretrained_lda:
        lda = pretrained_lda
    else:
        lda = train_lda(corpus, num_topics)

    topic_words = []

    for topic in lda.print_topics():
        topic_words.append(topic_to_words(topic[1]))

    topic_matrix = []

    #Get probability vector of each document in the list
    #Create matrix
    #---------------------------------------------------------------------------------
    for doc in docs_list:
        doc = corpus.doc_to_vec(doc)
        topic_vector = lda.get_document_topics(doc, minimum_probability=0)
        topic_vector = np.array([prob for _, prob in topic_vector])
        topic_matrix.append(topic_vector)

    return np.array(topic_matrix), topic_words

#================================================================================================================

def topics_to_clusters(topic_matrix: np.ndarray, doc_ids = None) -> tuple[list, list, list]:
    '''
    Takes a topic matrix and clusters the rows (documents)

    - clusters
    - strong_topics
    - labels
    '''

    if doc_ids is None:
        doc_ids = list(range(1, len(topic_matrix)+1))

    #Cluster probability vectors with KMeans
    #---------------------------------------------------------------------------------
    num_clusters = len(topic_matrix[0])
    clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
    clustering_model.fit(topic_matrix)

    #print(clustering_model.cluster_centers_)

    #Identify the main topic of each cluster
    #---------------------------------------------------------------------------------
    strong_topics = {}

    for i, center in enumerate(clustering_model.cluster_centers_):
        for j, topic in enumerate(center):
            if topic > 0.7:
                strong_topics[i] = j
                #print(f"Cluster {i:02} has strong topic {j:02} ({topic})")

    clusters = [[] for _ in range(num_clusters)]

    for i, label in enumerate(clustering_model.labels_):
        clusters[label].append(doc_ids[i])

    return clusters, strong_topics, clustering_model.labels_

#================================================================================================================

def visualize_clusters(topic_matrix: np.ndarray, labels: list):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(topic_matrix)

    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c = labels)
    ax.legend(*scatter.legend_elements())
    plt.show()
    
#==========================================================================================

if __name__ == "__main__":
    pass
    '''
    client = elasticsearch_client("credentials.json", "http_ca.crt")
    index_name = "test-index"
    collection_path = "collection"

    corpus = ScrollingCorpus(client, index_name, "counts.dict", "phrase_model.pkl")
    
    #queries = parse_queries("collection")
    #print(queries[0].text)
    #docs, doc_ids = query(client, index_name, queries[0])

    print(doc_ids)

    texts = docs_to_texts(docs)
    topic_matrix, _ = docs_to_topics(corpus, texts, 6, LdaModel.load("notebooks/lda.model"))
    clusters, _, labels = topics_to_clusters(topic_matrix, doc_ids)
    visualize_clusters(topic_matrix, labels)

    print(clusters)
    '''



