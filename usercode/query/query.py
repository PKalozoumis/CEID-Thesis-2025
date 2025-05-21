import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import Session, Query, ElasticDocument
from mypackage.storage import load_pickles
from rich.console import Console
from mypackage.clustering import visualize_clustering
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

console = Console()

#===============================================================================================================

if __name__ == "__main__":

    #Retrieval stage
    #-----------------------------------------------------------------------------------------------------------------
    #sess = Session("pubmed-index", base_path="..")
    sess = Session("pubmed-index", use="cache", cache_dir="../cache")
    query = Query(0, "What are the primary behaviours and lifestyle factors that contribute to childhood obesity", source=["summary", "article"], text_path="article", cache_dir="cache")
    #res = query.execute(sess)

    returned_docs = [
        ElasticDocument(sess, id=1923),
        ElasticDocument(sess, id=4355),
        ElasticDocument(sess, id=4166),
        ElasticDocument(sess, id=3611),
        ElasticDocument(sess, id=6389),
        ElasticDocument(sess, id=272),
        ElasticDocument(sess, id=2635),
        ElasticDocument(sess, id=2581),
        ElasticDocument(sess, id=372),
        ElasticDocument(sess, id=6415)
    ]

    #Load the clusters corresponding to the retrieved documents
    pkl_list = load_pickles(sess, "../experiments/pubmed-index/pickles/default", docs = [doc.id for doc in returned_docs])

    #-----------------------------------------------------------------------------------------------------------------

    #Encode the query
    if not os.path.exists("query.npy"):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    else:
        model = None

    query.load_vector(model)
        
    clusters = []
    doc_labels = []

    for doc_number, pkl in enumerate(pkl_list):
        for cluster in pkl.clustering:
            if cluster.label > -1:
                clusters.append(cluster)
                doc_labels.append(doc_number)

    #visualize_clustering(clusters, doc_labels, show=True)

    #Find the similarity to each cluster centroid
    sim = cosine_similarity([cluster.vector for cluster in clusters], query.vector.reshape((1,-1)))
    sorted_clusters = [list(x) for x in sorted(zip(sim, clusters, doc_labels), reverse=True)]

    #Mark top k clusters
    k = 7
    for i in range(k):
        sorted_clusters[i][2] = 11
        print(sorted_clusters[i][1].doc.id)

    visualize_clustering([t[1] for t in sorted_clusters], [t[2] for t in sorted_clusters], show=True, return_legend=True, extra_vector=query.vector)
    

    