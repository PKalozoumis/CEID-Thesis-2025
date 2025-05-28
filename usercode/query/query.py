import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import Session, ElasticDocument
from mypackage.clustering.metrics import cluster_stats
from mypackage.helper import panel_print
from mypackage.query import Query
from mypackage.summarization import SummarySegment, Summarizer
from mypackage.cluster_selection import SelectedCluster, RelevanceEvaluator, cluster_retrieval, context_expansion, print_candidates

from rich.console import Console
from sentence_transformers import SentenceTransformer

from matplotlib import pyplot as plt
from sentence_transformers import CrossEncoder
from rich.rule import Rule
from rich.pretty import Pretty
from rich.live import Live
import argparse

from dataclasses import dataclass, field

console = Console()

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

    console.print(f"\n[green]Query:[/green] {query.text}\n")

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

    #-----------------------------------------------------------------------------------------------------------------

    #Encode the query
    if not os.path.exists("query.npy"):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    else:
        model = None
    query.load_vector(model)

    #Retrieve clusters from docs
    selected_clusters = cluster_retrieval(sess, returned_docs, query)

    panel_print([f"Cluster [green]{cluster.id}[/green] with score [cyan]{cluster.sim:.3f}[/cyan]" for cluster in selected_clusters], title="Retrieved clusters based on cosine similarity")

    #Print cluster stats
    if args.stats:
        panel_print([Pretty(cluster_stats(cluster)) for cluster in selected_clusters], title="Cluster Stats")

    evaluator = RelevanceEvaluator(query, CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2'))

    #Calculate the cross-encoder scores
    #-----------------------------------------------------------------------------------------------------------------
    os.makedirs("cross_scores", exist_ok=True)
    for cluster in selected_clusters:
        cluster.evaluator = evaluator
        if not os.path.exists(f"cross_scores/{cluster.id}.pkl"):
            cluster.evaluate_chains()
            cluster.store_scores("cross_scores")
        else:
            cluster.load_scores("cross_scores")

    panel_print([f"Cluster {cluster.id} score: [cyan]{cluster.cross_score:.3f}[/cyan]" for cluster in selected_clusters], title="Cross-encoder scores of the selected clusters")

    if args.c != -1:
        selected_clusters = [selected_clusters[args.c]]

    #-----------------------------------------------------------------------------------------------------------------

    for focused_cluster in selected_clusters:
        focused_cluster: SelectedCluster

        #Print cluster chains
        if args.print:
            focused_cluster.print()

        #Let's evaluate chains
        print_candidates(focused_cluster)

        #Context Expansion
        #-----------------------------------------------------------------------------------------------------------------
        context_expansion(focused_cluster)
        focused_cluster.merge_candidates()
        print_candidates(focused_cluster)
        panel_print(focused_cluster.text, title=f"Text (size = {len(focused_cluster.text.split())})")

        #Summarization
        #-----------------------------------------------------------------------------------------------------------------
        #Create summary segments out of the selected clusters
        #For now we'll play with just one cluster
        seg = SummarySegment([focused_cluster], focused_cluster.cluster.doc)
        summarizer = Summarizer()