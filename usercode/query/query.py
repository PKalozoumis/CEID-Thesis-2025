import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import Session, Query, ElasticDocument
from mypackage.storage import load_pickles
from rich.console import Console
from mypackage.clustering import visualize_clustering, ChainCluster
from mypackage.clustering.metrics import cluster_stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from operator import methodcaller
from matplotlib import pyplot as plt
from mypackage.helper import panel_print
from sentence_transformers import CrossEncoder
from rich.rule import Rule
from rich.padding import Padding
from rich.pretty import Pretty
import numpy as np
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
from mypackage.sentence import SentenceChain
from rich.live import Live
from rich.panel import Panel
import argparse
from mypackage.summarization import evaluate_summary_relevance
import json

from dataclasses import dataclass, field

console = Console()

@dataclass
class SelectedCluster():

    @dataclass
    class ScoredChain():
        chain: SentenceChain
        score: float

        def __getattr__(self, name):
            return getattr(self.chain, name)

    cluster: ChainCluster
    query: Query
    score: float #Similarity to query
    scored_chains: list[ScoredChain] = field(default=None)

    #---------------------------------------------------------------------------

    def print(self):
        group = []
        for i, chain in enumerate(self.cluster.chains):
            group.append(Pretty(f"{i:02}. Chain {chain}"))
            group.append(Rule())
            group.append(Padding(chain.text, pad=(0,0,2,0)))

        panel_print(group)

    #---------------------------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self.cluster, name)
    
    #---------------------------------------------------------------------------
    
    def evaluate_chains(self, model: CrossEncoder) -> 'SelectedCluster':
        self.chain_scores = []

        scores = model.predict([(self.query.text, c.text) for c in self.chains])
        for i, score in enumerate(scores):
            sc = np.round(score, decimals=3)
            self.chain_scores.append(sc)
        
        #Sort chains
        self.scored_chains = [self.ScoredChain(chain, score) for score, chain in sorted(zip(scores, self.cluster), reverse=True)]
        return self
    
    #---------------------------------------------------------------------------
    
    def __len__(self):
        return len(self.cluster)
    
    #---------------------------------------------------------------------------
        
    def __getitem__(self, i: int) -> SentenceChain:
        return self.chains[i]
    
    #---------------------------------------------------------------------------
        
    def __iter__(self):
        return iter(self.chains)
    
#==============================================================================================================

def store_summary(cluster: SelectedCluster, summary: str, args):
    os.makedirs("generated_summaries", exist_ok=True)

    with open(f"generated_summaries/cluster_{cluster.id}.json", "w") as f:
        json.dump({
            'generated_with': "llm" if args.m == "llm" else "transformer",
            'summary': summary
        }, f)

#==============================================================================================================

def load_summary(cluster: SelectedCluster):
    if not os.path.exists(f"generated_summaries/cluster_{cluster.id}.json"):
        return None

    with open(f"generated_summaries/cluster_{cluster.id}.json", "r") as f:
        return json.load(f)['summary']

#===============================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", action="store", type=int, default=0, help="The cluster number")
    parser.add_argument("-m", action="store", type=str, default="sus", help="The model to use", choices=["sus", "llm"])
    parser.add_argument("-s", action="store", type=str, default="thres", help="Best cluster selection method", choices=["topk", "thres"])
    parser.add_argument("--print", action="store_true", default=False, help="Print the cluster's text")
    parser.add_argument("--stats", action="store_true", default=False, help="Print cluster stats")
    parser.add_argument("--cache", dest="cache", action="store_true", default=True, help="Try to load summary from cache")
    parser.add_argument("--no-cache", dest="cache", action="store_false", help="Try to load summary from cache")

    args = parser.parse_args()

    console.print(Pretty(args))

    #Retrieval stage
    #-----------------------------------------------------------------------------------------------------------------
    #sess = Session("pubmed-index", base_path="..")
    sess = Session("pubmed-index", use="cache", cache_dir="../cache")
    query = Query(0, "What are the primary behaviours and lifestyle factors that contribute to childhood obesity", source=["summary", "article"], text_path="article", cache_dir="cache")
    #res = query.execute(sess)

    console.print(f"Query: {query.text}")

    returned_docs = [
        ElasticDocument(sess, id=1923, text_path="article"),
        ElasticDocument(sess, id=4355, text_path="article"),
        ElasticDocument(sess, id=4166, text_path="article"),
        ElasticDocument(sess, id=3611, text_path="article"),
        ElasticDocument(sess, id=6389, text_path="article"),
        ElasticDocument(sess, id=272, text_path="article"),
        ElasticDocument(sess, id=2635, text_path="article"),
        ElasticDocument(sess, id=2581, text_path="article"),
        ElasticDocument(sess, id=372, text_path="article"),
        ElasticDocument(sess, id=6415, text_path="article")
    ]

    #Load the clusters corresponding to the retrieved documents
    pkl_list = load_pickles(sess, "../experiments/pubmed-index/pickles/default", docs = returned_docs)

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
    #Select best clusters
    #==========================================================================================================================
    sim = cosine_similarity([cluster.vector for cluster in clusters], query.vector.reshape((1,-1)))
    sorted_clusters = [list(x) for x in sorted(zip(map(methodcaller("__getitem__", 0), sim), clusters, doc_labels), reverse=True)]

    selected_clusters = []
    selected_clusters: list[SelectedCluster]

    if args.s == "topk":
        #Mark top k clusters
        k = 7
        for i in range(k):
            sorted_clusters[i][2] = 11
            #print(sorted_clusters[i][0])
            console.print((sorted_clusters[i][1].doc.id, sorted_clusters[i][0]))
            selected_clusters.append(SelectedCluster(sorted_clusters[i][1], query, sorted_clusters[i][0]))
    else:
        thres = 0.5
        for cluster in sorted_clusters:
            if cluster[0] > thres:
                cluster[2] = 11
                console.print((cluster[1].doc.id, cluster[0]))
                selected_clusters.append(SelectedCluster(cluster[1], query, cluster[0]))
            else:
                break

    #Visualize
    #===============================================================================================================
    plt.close('all')
    #visualize_clustering([t[1] for t in sorted_clusters], [t[2] for t in sorted_clusters], show=True, return_legend=True, extra_vector=query.vector)

    if args.print:
        selected_clusters[args.c].print()

    #Let's evaluate chains
    #===============================================================================================================
    cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2')
    cross_encoder_threshold = 3

    for cluster in selected_clusters:
        cluster.evaluate_chains(cross_model)

    #What are the cluster sizes:
    '''
    for i, cluster in enumerate(selected_clusters):
        console.print(f"Cluster {i:02}")
        console.print(Rule())
        console.print(Pretty(cluster_stats(cluster.cluster)))
        console.print()
    '''

    #What happens if I combine
    '''
    combined_text = selected_clusters[test_cluster][1][6].text + selected_clusters[test_cluster][1][7].text + selected_clusters[test_cluster][1][8].text
    print(combined_text)
    print(cross_model.predict((query.text, combined_text)))
    '''

    if args.cache:
        summary = load_summary(selected_clusters[args.c])

    if not args.cache or summary is None:

        if args.m == "sus":
            #Let's try summarization of the entire cluster
            tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")
            summ_model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed")
            #panel_print(selected_clusters[args.c].text)
            inputs = tokenizer(selected_clusters[args.c].text, return_tensors='pt', truncation=True, max_length=4096)

            #print(selected_clusters[test_cluster][1].text)

            prediction = summ_model.generate(**inputs)
            prediction = tokenizer.batch_decode(prediction)

            panel_print(prediction)
            summary = prediction

        #----------------------------------------------------------------------------------------

        elif args.m == "llm":
            from mypackage.llm import summarize
            
            #What will the llm do?
            #......what will I do
            full_text = ""
            removed_json = False
            with Live(panel_print(return_panel=True), refresh_per_second=10) as live:
                for fragment in summarize(query.text, selected_clusters[args.c].text):

                    full_text += fragment

                    if fragment.endswith("}"):
                        full_text = full_text[:-2]

                    if not removed_json and full_text.startswith("{\"summary\": \""):
                        full_text = full_text[13:]
                        removed_json = True
                    else:
                        live.update(panel_print(full_text, return_panel=True))

            summary = full_text

        store_summary(selected_clusters[args.c], summary, args)

    #Evaluating the generated summary
    evaluate_summary_relevance(cross_model, summary, query.text)