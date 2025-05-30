import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import Session, ElasticDocument, Document
from mypackage.clustering.metrics import cluster_stats
from mypackage.helper import panel_print
from mypackage.query import Query
from mypackage.summarization import SummarySegment, Summarizer
from mypackage.cluster_selection import SelectedCluster, RelevanceEvaluator, cluster_retrieval, context_expansion, print_candidates
from mypackage.llm import LLMSession, merge_summaries
from mypackage.sentence import doc_to_sentences

from rich.console import Console
from rich.live import Live
from rich.rule import Rule
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from rich.pretty import Pretty
import argparse
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import time

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
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    query.load_vector(sentence_model)

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

    t = time.time()

    clusters_per_doc = defaultdict(list)

    for focused_cluster in selected_clusters:
        focused_cluster: SelectedCluster

        clusters_per_doc[focused_cluster.cluster.doc].append(focused_cluster)

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
        #panel_print(focused_cluster.text, title=f"Text (size = {len(focused_cluster.text.split())})")

    print(time.time() - t)

    #Summarization
    #-----------------------------------------------------------------------------------------------------------------    
    segments: list[SummarySegment] = []

    for doc, cluster_list in clusters_per_doc.items():
        seg = SummarySegment(cluster_list, doc)
        seg.load_summary()
        segments.append(seg)

        
        panel_print([seg.text, Rule(), seg.summary.text], title=f"Summary of {seg.doc.id} ({seg.created_with})")

    llm = LLMSession("meta-llama-3.1-8b-instruct")

    #panel_print(segments[0].text, title=f"Text (size = {len(segments[0].text.split())})")
    #summarizer = Summarizer(query)
    #summarizer.summarize_segments(segments)

    #Creating citations
    #------------------------------------------------------------------------------
    selected_segment = segments[0]

    #Transform the summary into embeddings
    doc_to_sentences(selected_segment.summary, sentence_model, sep=".")
    console.print([s.text for s in selected_segment.summary.sentences])

    #See the chains of the segment
    flat_chains = selected_segment.flat_chains()
    panel_print([f"{x.index:03}. {x.text}\n" for x in flat_chains], title="Chains")

    #Calculate cosine similarity between each sentence of the summary and the text
    #(may need to improve down the line)
    sims = cosine_similarity(selected_segment.summary_matrix(), flat_chains)
    max_sim = np.argmax(sims, axis=1)
    console.print(max_sim)

    selected_segment.citations = [None]*len(selected_segment.summary.sentences)
    for summary_sentence_index, chain_pos in enumerate(max_sim):


    #Add citations to the summary
    

    #Retrieve fragments of text from the llm and add them to the full text
    #------------------------------------------------------------------------------

    if False:
        full_text = ""
        with Live(panel_print(return_panel=True), refresh_per_second=10) as live:
            for fragment in merge_summaries(llm, segments, query.text):
                full_text += fragment
                live.update(panel_print(full_text, return_panel=True))

    
    