import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import Session, ElasticDocument, Document
from mypackage.clustering.metrics import cluster_stats
from mypackage.helper import panel_print
from mypackage.query import Query
from mypackage.summarization import Summarizer, SummaryUnit
from mypackage.cluster_selection import SelectedCluster, RelevanceEvaluator, cluster_retrieval, context_expansion, print_candidates
from mypackage.llm import LLMSession
from mypackage.sentence import doc_to_sentences

from sentence_transformers import SentenceTransformer, CrossEncoder
import argparse
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from itertools import chain

from rich.pretty import Pretty
from rich.console import Console
from rich.live import Live
from rich.rule import Rule
from rich.padding import Padding
from rich.tree import Tree

from classes import Message, Arguments

console = Console()

#===============================================================================================================

def query_function(query_str: str, *, args: Arguments = None, base_path: str = "..", sse_format: bool = False, console_messages: bool = False):
    def message_sender(msg: Message):
        return msg.to_sse() if sse_format else msg.to_json(string=True)
    
    console.print(f"[green]Console messages[/green]: {console_messages}")

    if args is None:
        args = Arguments()
    
    times = defaultdict(float)

    #Retrieval stage
    #-----------------------------------------------------------------------------------------------------------------
    sess = Session("pubmed", base_path=base_path, use="cache", cache_dir=f"{base_path}/cache")
    query = Query(0, "What are the primary behaviours and lifestyle factors that contribute to childhood obesity", source=["summary", "article"], text_path="article")
    
    times['elastic'] = time.time()
    #res = query.execute(sess)
    times['elastic'] = time.time() - times['elastic']

    if console_messages:
        yield message_sender(Message("query", query.text))
    #console.print(f"\n[green]Query:[/green] {query.text}\n")

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
        ElasticDocument(sess, id=1106, text_path="article")
    ]

    #-----------------------------------------------------------------------------------------------------------------

    #Encode the query
    times['query_encode'] = time.time()
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    query.load_vector(sentence_model)
    times['query_encode'] = time.time() - times['query_encode']

    #Retrieve clusters from docs
    times['cluster_retrieval'] = time.time()
    selected_clusters = cluster_retrieval(sess, returned_docs, query, base_path=base_path)
    times['cluster_retrieval'] = time.time() - times['cluster_retrieval']

    if console_messages:
        yield message_sender(Message("cosine_sim", [{'id': cluster.id, 'sim': cluster.sim} for cluster in selected_clusters]))

    #panel_print([f"[green]{i:02}.[/green] Cluster [green]{cluster.id}[/green] with score [cyan]{cluster.sim:.3f}[/cyan]" for i, cluster in enumerate(selected_clusters)], title="Retrieved clusters based on cosine similarity")

    #Print cluster stats
    if args.stats:
        if console_messages:
            yield message_sender(Message("console", {
                'type': 'cluster_stats',
                'data': [cluster_stats(cluster) for cluster in selected_clusters]
            }))
        #panel_print([Pretty(cluster_stats(cluster)) for cluster in selected_clusters], title="Cluster Stats")

    evaluator = RelevanceEvaluator(query, CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2'))

    #Calculate the cross-encoder scores
    #-----------------------------------------------------------------------------------------------------------------
    os.makedirs("cross_scores", exist_ok=True)
    for cluster in selected_clusters:
        times[f'cross_score_{cluster.id}'] = time.time()
        cluster.evaluator = evaluator
        cluster.evaluate_chains()
        cluster.store_scores("cross_scores")
        times[f'cross_score_{cluster.id}'] = time.time() - times[f'cross_score_{cluster.id}']

    
    panel_print([f"Cluster [green]{cluster.id}[/green] score: [cyan]{cluster.cross_score:.3f}[/cyan]" for cluster in selected_clusters], title="Cross-encoder scores of the selected clusters")

    if args.c != -1:
        selected_clusters = [selected_clusters[args.c]]
    

    #-----------------------------------------------------------------------------------------------------------------
    cross_scores = []

    yield message_sender(Message("fragment", "Expanding context..."))

    for focused_cluster in selected_clusters:
        focused_cluster: SelectedCluster

        console.print(Rule(title=f"Cluster {focused_cluster.id}", align="center"))
        cross_scores.append(focused_cluster.cross_score)

        #Print cluster chains
        if args.print:
            focused_cluster.print()

        #Let's evaluate chains
        print_candidates(focused_cluster)

        #Context Expansion
        #-----------------------------------------------------------------------------------------------------------------
        times[f'context_expansion_{focused_cluster.id}'] = time.time()
        context_expansion(focused_cluster, threshold=args.cet)
        focused_cluster.filter_candidates().merge_candidates()
        print_candidates(focused_cluster, title=f"Merged candidates for cluster {focused_cluster.id}")
        #panel_print(focused_cluster.text, title=f"Text (size = {len(focused_cluster.text.split())})")
        times[f'context_expansion_{focused_cluster.id}'] = time.time() - times[f'context_expansion_{focused_cluster.id}']

    panel_print([f"Cluster [green]{cluster.id}[/green] score: [cyan]{cross_scores[i]}[/cyan] -> [cyan]{cluster.cross_score:.3f}[/cyan] ([green]+{round(cluster.cross_score - cross_scores[i], 3):.3f}[/green])" for i, cluster in enumerate(selected_clusters)], title="Cross-encoder scores of the selected clusters after context expansion")
    panel_print([f"Cluster [green]{cluster.id}[/green] score: [cyan]{cluster.selected_candidate_cross_score:.3f}[/cyan]" for i, cluster in enumerate(selected_clusters)], title="Cross-encoder scores (only selected candidates considered)")

    #Summarization
    #-----------------------------------------------------------------------------------------------------------------
    #Print text
    unit = SummaryUnit(selected_clusters, sorting_method=args.csm)
    unit.pretty_print(show_added_context=True, show_chain_indices=True)
    panel_print(unit.text)
    
    #Summarize
    if args.summ:
        is_first_fragment = True

        llm = LLMSession("meta-llama-3.1-8b-instruct")

        summarizer = Summarizer(query, llm=llm)
        times['summary_time'] = time.time()
        times['summary_response_time'] = time.time()

        #Generate the fragments
        stop_dict = {'stop': False, 'stopped': False}
        for stream, fragment in summarizer.summarize(unit, stop_dict):
            try:
                if is_first_fragment:
                    times['summary_response_time'] = time.time() - times['summary_response_time']
                    is_first_fragment = False
                if stop_dict['stopped']:
                    break
                else:
                    yield message_sender(Message("fragment", fragment))
            except GeneratorExit:
                stop_dict['stop'] = True
                console.print("[red]Client disconnected[/red]")
            except BrokenPipeError:
                stop_dict['stop'] = True
                console.print("[red]Client disconnected (broken pipe)[/red]")

        times['summary_time'] = time.time() - times['summary_time']

    #Print times
    #------------------------------------------------------------------------------
    times = defaultdict(float, {k:round(v, 3) for k,v in times.items()})

    tree = Tree(f"[green]Total time: [cyan]{sum(times.values()):.3f}s[/cyan]")
    tree.add(f"[green]Elasticsearch time: [cyan]{times['elastic']:.3f}s[/cyan]")
    tree.add(f"[green]Query encoding: [cyan]{times['query_encode']:.3f}s[/cyan]")
    tree.add(f"[green]Cluster retrieval: [cyan]{times['cluster_retrieval']:.3f}s[/cyan]")

    score_tree = tree.add(f"[green]Cross-scores: [cyan]{sum(v for k,v in times.items() if k.startswith('cross_score')):.3f}s[/cyan]")
    for k,v in times.items():
        if k.startswith('cross_score'):
            score_tree.add(f"[green]Cluster {k[12:]}: [cyan]{v:.3f}s[/cyan]")

    context_tree = tree.add(f"[green]Context expansion: [cyan]{sum(v for k,v in times.items() if k.startswith('context_expansion')):.3f}s[/cyan]")
    for k,v in times.items():
        if k.startswith('context_expansion'):
            context_tree.add(f"[green]Cluster {k[18:]}: [cyan]{v:.3f}s[/cyan]")

    summary_tree = tree.add(f"[green]Summarization[/green]: [cyan]{times['summary_time']}s[/cyan]")
    summary_tree.add(f"[green]Response time[/green]: [cyan]{times['summary_response_time']:.3f}s[/cyan]")


    console.print(tree)

    if not stop_dict['stop']:
        yield message_sender(Message("end", 1))