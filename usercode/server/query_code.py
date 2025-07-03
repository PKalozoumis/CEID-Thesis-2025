import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import Session, ElasticDocument, Document
from mypackage.clustering.metrics import cluster_stats
from mypackage.helper import panel_print, rich_console_text
from mypackage.query import Query
from mypackage.summarization import Summarizer, SummaryUnit
from mypackage.cluster_selection import SelectedCluster, RelevanceEvaluator, cluster_retrieval, context_expansion, context_expansion_generator, print_candidates
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
    
    yield message_sender(Message("info", "Retrieving documents..."))
    times['elastic'] = time.time()
    #res = query.execute(sess)
    times['elastic'] = time.time() - times['elastic']
    yield message_sender(Message('time', {'elastic': times['elastic']}))

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
    yield message_sender(Message("info", "Encoding query..."))
    times['query_encode'] = time.time()
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    query.load_vector(sentence_model)
    times['query_encode'] = time.time() - times['query_encode']
    yield message_sender(Message('time', {'query_encode': times['query_encode']}))

    #Retrieve clusters from docs
    yield message_sender(Message("info", "Extracting relevant information..."))
    times['cluster_retrieval'] = time.time()
    selected_clusters = cluster_retrieval(sess, returned_docs, query, base_path=base_path)
    times['cluster_retrieval'] = time.time() - times['cluster_retrieval']
    yield message_sender(Message('time', {'cluster_retrieval': times['cluster_retrieval']}))

    if console_messages:
        yield message_sender(Message("cosine_sim", [{'id': cluster.id, 'sim': cluster.sim} for cluster in selected_clusters]))

    #Send cluster stats to client
    if args.stats:
        if console_messages:
            yield message_sender(Message("cluster_stats", [cluster_stats(cluster) for cluster in selected_clusters]))

    evaluator = RelevanceEvaluator(query, CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2'))

    #Calculate the cross-encoder scores
    #-----------------------------------------------------------------------------------------------------------------
    os.makedirs("cross_scores", exist_ok=True)
    for cluster in selected_clusters:
        key = f'cross_score_{cluster.id}'

        times[key] = time.time()
        cluster.evaluator = evaluator
        cluster.evaluate_chains()
        cluster.store_scores("cross_scores")
        times[key] = time.time() - times[key]
        yield message_sender(Message('time', {key: times[key]}))

    #Send cross-encoder scores to client
    if console_messages:
        yield message_sender(Message("cross_scores", [{'id': cluster.id, 'cross_score': cluster.cross_score} for cluster in selected_clusters]))

    if args.c != -1:
        selected_clusters = [selected_clusters[args.c]]

    #-----------------------------------------------------------------------------------------------------------------
    cross_scores = []

    yield message_sender(Message("info", "Expanding relevant information..."))

    for focused_cluster in selected_clusters:
        focused_cluster: SelectedCluster
        key = f'context_expansion_{focused_cluster.id}'

        yield message_sender(Message("context_expansion_progress", focused_cluster.id))

        cross_scores.append(focused_cluster.cross_score)

        #Print cluster chains
        if args.print:
            focused_cluster.print()

        #Let's evaluate chains
        res = print_candidates(focused_cluster, return_text=True)
        if console_messages:
            yield message_sender(Message("ansi_text", res))

        #Context Expansion
        #-----------------------------------------------------------------------------------------------------------------
        times[key] = time.time()
        yield message_sender(Message('time', {key: times[key]}))

        if console_messages:
            for text in context_expansion_generator(focused_cluster, threshold=args.cet):
                yield message_sender(Message("ansi_text", text))
        else:
            context_expansion(focused_cluster, threshold=args.cet)

        #Keep candidates that are above a threshold
        focused_cluster.filter_candidates().merge_candidates()

        #Send final result of expansion to client
        res = print_candidates(focused_cluster, title=f"Merged candidates for cluster {focused_cluster.id}", return_text=True)
        if console_messages:
            yield message_sender(Message("ansi_text", res))

        times[key] = time.time() - times[key]
        yield message_sender(Message('time', {key: times[key]}))

    #Send the score comparisons to client
    #-----------------------------------------------------------------------------------------------------------------
    if console_messages:
        yield message_sender(Message("cross_scores_2", [
            {
                'id': cluster.id,
                'original_score': cross_scores[i],
                'new_score': cluster.cross_score,
                'selected_score': cluster.selected_candidate_cross_score
            }
            for i, cluster in enumerate(selected_clusters)
        ]))

    #Summarization
    #-----------------------------------------------------------------------------------------------------------------
    #Print text
    unit = SummaryUnit(selected_clusters, sorting_method=args.csm)
    
    if console_messages:
        res = unit.pretty_print(show_added_context=True, show_chain_indices=True, return_text=True)
        yield message_sender(Message("ansi_text", res))

        res = rich_console_text(panel_print(unit.text, return_panel=True))
        yield message_sender(Message("ansi_text", res))
    
    #Summarize
    if args.summ:
        is_first_fragment = True

        yield message_sender(Message("info", "Summarizing..."))
        llm = LLMSession("meta-llama-3.1-8b-instruct")

        summarizer = Summarizer(query, llm=llm)
        times['summary_time'] = time.time()
        times['summary_response_time'] = time.time()

        #Generate the fragments
        stop_dict = {'stop': False, 'stopped': False}
        for fragment, citation in summarizer.summarize(unit, stop_dict):
            try:
                if is_first_fragment:
                    times['summary_response_time'] = time.time() - times['summary_response_time']
                    yield message_sender(Message('time', {'summary_response_time': times['summary_response_time']}))
                    is_first_fragment = False
                if stop_dict['stopped']:
                    break
                else:
                    if citation is not None:
                        '''
                        #Find document that the citation refers to
                        doc = next(doc for doc in returned_docs if doc.id == citation['doc'])
                        chains = doc.sentences[0].parent_chain.parent_cluster.clustering_context.chains

                        #We need to turn the chain index into a sentence index
                        citation['start'] = chains[citation['start']].offset_range.start
                        citation['end'] = chains[citation['end']].offset_range.stop
                        '''

                        yield message_sender(Message("fragment_with_citation", {'fragment': fragment, 'citation': citation}))
                    else:
                        yield message_sender(Message("fragment", fragment))

            except GeneratorExit:
                stop_dict['stop'] = True
                console.print("[red]Client disconnected[/red]")
            except BrokenPipeError:
                stop_dict['stop'] = True
                console.print("[red]Client disconnected (broken pipe)[/red]")

        times['summary_time'] = time.time() - times['summary_time']
        yield message_sender(Message('time', {'summary_time': times['summary_time']}))

    if not stop_dict['stop']:
        yield message_sender(Message("end", 1))