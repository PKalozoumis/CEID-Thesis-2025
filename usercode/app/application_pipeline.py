import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import Session, ElasticDocument
from mypackage.clustering.metrics import cluster_stats
from mypackage.query import Query
from mypackage.summarization import Summarizer, SummaryUnit
from mypackage.cluster_selection import SelectedCluster, RelevanceEvaluator, cluster_retrieval, context_expansion, context_expansion_generator, print_candidates
from mypackage.cluster_selection.metrics import single_document_cross_score
from mypackage.llm import LLMSession
from mypackage.storage import DatabaseSession, MongoSession, PickleSession

from application_classes import Arguments

from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import time
from flask_socketio import SocketIO
from rich.console import Console

console = Console()
message_sender = None

_console_width = None

#===============================================================================================================

def retrieval_stage(sess, query, *, query_str: str, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 1 of the pipeline
    '''
    
    message_sender("info", "Retrieving documents...")

    times['elastic'] = time.time()
    #res = query.execute(sess)
    times['elastic'] = time.time() - times['elastic']

    message_sender("time", {'elastic': times['elastic']})

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

    return returned_docs

#===============================================================================================================

def encode_query(query: Query, *, query_str: str, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 2 of the pipeline
    '''
    message_sender('info', "Encoding query...")

    times['query_encode'] = time.time()
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    query.encode(sentence_model)
    times['query_encode'] = time.time() - times['query_encode']

    message_sender('time', {'query_encode': times['query_encode']})

#===============================================================================================================

def retrieve_clusters(sess, db, returned_docs, query, *, keep_cluster: int, query_str: str, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 3 of the pipeline
    '''
    message_sender('info', "Extracting relevant information...")

    times['cluster_retrieval'] = time.time()
    selected_clusters = cluster_retrieval(sess, db, returned_docs, query, base_path=base_path, experiment=args.x)
    times['cluster_retrieval'] = time.time() - times['cluster_retrieval']
    
    message_sender('time', {'cluster_retrieval': times['cluster_retrieval']})

    if args.print:
        message_sender("cosine_sim", [{'id': cluster.id, 'sim': float(cluster.sim)} for cluster in selected_clusters])

    #Send cluster stats to client
    if args.stats:
        if args.print:
            message_sender("cluster_stats", [cluster_stats(cluster) for cluster in selected_clusters])

    #Restrict the retrieved clusters based on client parameter
    if keep_cluster != -1:
        selected_clusters = [selected_clusters[keep_cluster]]

    return selected_clusters

#===============================================================================================================

def calculate_cross_scores(query, selected_clusters: list[SelectedCluster], *, query_str: str, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 4 of the pipeline
    '''
    evaluator = RelevanceEvaluator(query, CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2'))

    for cluster in selected_clusters:
        key = f'cross_score_{cluster.id}'
        times[key] = time.time()

        cluster.evaluator = evaluator
        cluster.evaluate_chains()
        
        times[key] = time.time() - times[key]
        message_sender('time', {key: times[key]})

    #Send cross-encoder scores to client
    if args.print:
        message_sender("cross_scores", [{'id': cluster.id, 'cross_score': float(cluster.cross_score)} for cluster in selected_clusters])

    return evaluator

#===============================================================================================================

def expand_context(selected_clusters: list[SelectedCluster], *, query_str: str, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 5 of the pipeline
    '''
    cross_scores = []

    message_sender("info", "Expanding relevant information...")

    for focused_cluster in selected_clusters:
        focused_cluster: SelectedCluster
        key = f'context_expansion_{focused_cluster.id}'

        message_sender("context_expansion_progress", focused_cluster.id)

        cross_scores.append(focused_cluster.cross_score)

        #Let's evaluate chains
        res = print_candidates(focused_cluster, return_text=True)
        if args.print:
            message_sender("ansi_text", res)

        #Context Expansion
        #-----------------------------------------------------------------------------------------------------------------
        times[key] = time.time()
        message_sender('time', {key: times[key]})

        if args.print:
            for text in context_expansion_generator(focused_cluster, threshold=args.cet):
                message_sender("ansi_text", text)
        else:
            context_expansion(focused_cluster, threshold=args.cet)

        #Keep candidates that are above a threshold
        #focused_cluster.filter_candidates().merge_candidates()
        focused_cluster.filter_and_merge_candidates()

        #Send final result of expansion to client
        res = print_candidates(focused_cluster, title=f"Merged candidates for cluster {focused_cluster.id}", return_text=True, current_state_only=True)
        if args.print:
            message_sender("ansi_text", res)

        times[key] = time.time() - times[key]
        message_sender('time', {key: times[key]})

    #Send the score comparisons to client
    #-----------------------------------------------------------------------------------------------------------------
    if args.print:
        message_sender("cross_scores_2", [
            {
                'id': cluster.id,
                'original_score': float(cross_scores[i]),
                'new_score': float(cluster.cross_score),
                'selected_score': float(cluster.selected_candidate_cross_score)
            }
            for i, cluster in enumerate(selected_clusters)
        ])

#===============================================================================================================

def summarization_stage(query: Query, selected_clusters: list[SelectedCluster], stop_dict, *, query_str: str, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 6 of the pipeline
    '''
    #The text to be summarized
    unit = SummaryUnit(selected_clusters, sorting_method=args.csm)
    
    if args.print:
        res = unit.pretty_print(show_added_context=True, show_chain_indices=True, return_text=True, console_width=_console_width)
        message_sender("ansi_text", res)

    if args.summ:
        is_first_fragment = True

        message_sender("info", "Summarizing...")
        llm = LLMSession.create(server_args.llm_backend, "meta-llama-3.1-8b-instruct", api_host=server_args.host)

        summarizer = Summarizer(query, llm=llm)
        times['summary_time'] = time.time()
        times['summary_response_time'] = time.time()

        #Generate the fragments
        for fragment, citation in summarizer.summarize(unit, stop_dict, cache_prompt=True):
            
            if is_first_fragment:
                times['summary_response_time'] = time.time() - times['summary_response_time']
                message_sender('time', {'summary_response_time': times['summary_response_time']})
                message_sender("fragment", fragment)
                is_first_fragment = False
            else:
                if citation is not None:
                    message_sender("fragment_with_citation", {'fragment': fragment, 'citation': citation})
                else:
                    message_sender("fragment", fragment)

        times['summary_time'] = time.time() - times['summary_time']
        message_sender('time', {'summary_time': times['summary_time']})

#===============================================================================================================

def pipeline(query_str: str, stop_dict, *, args: Arguments = None, server_args, base_path: str = "..", socket: SocketIO, console_width: int):

    global message_sender, _console_width

    if args is None:
        args = Arguments()

    #Create a sender function that is bound to specific namespace. Saves time
    message_sender = lambda event, data: socket.emit(event, data, namespace="/query")

    _console_width = console_width
    times = defaultdict(float)
    kwargs = {'query_str': query_str, 'args': args, 'base_path': base_path, 'times': times, 'server_args': server_args}

    #Session initialization
    #--------------------------------------------------------------------------
    sess = Session("pubmed", base_path=base_path, use="cache", cache_dir=f"{base_path}/cache")
    query = Query(0, "What are the primary behaviours and lifestyle factors that contribute to childhood obesity", source=["summary", "article"], text_path="article")
    
    #Select database
    if server_args.db == "pickle":
        db = PickleSession(f"{base_path}/experiments/{sess.index_name}/pickles", args.experiment)
    else:
        db = MongoSession(db_name=f"experiments_{sess.index_name}", collection=args.experiment)

    #Pipeline stages pre-summarization
    #--------------------------------------------------------------------------
    returned_docs = retrieval_stage(sess, query, **kwargs)
    encode_query(query, **kwargs)
    selected_clusters = retrieve_clusters(sess, db, returned_docs, query, keep_cluster=args.c, **kwargs)
    evaluator = calculate_cross_scores(query, selected_clusters, **kwargs)
    expand_context(selected_clusters, **kwargs)

    #Evaluation
    #--------------------------------------------------------------------------
    doc = returned_docs[0]
    #single_document_cross_score(doc, [sc for sc in selected_clusters if sc.doc.id == doc.id], evaluator)

    #Summarization
    #--------------------------------------------------------------------------
    summarization_stage(query, selected_clusters, stop_dict, **kwargs)

    message_sender("end", {'status': 0})
    db.close()