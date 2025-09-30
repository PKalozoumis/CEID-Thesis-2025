import os
import sys
sys.path.append(os.path.abspath("../.."))

from application_helper import Arguments

from sentence_transformers import SentenceTransformer
from collections import defaultdict
import time
from flask_socketio import SocketIO
from rich.console import Console

from mypackage.elastic import Session
from mypackage.clustering.metrics import cluster_stats
from mypackage.query import Query
from mypackage.summarization import Summarizer, SummaryUnit
from mypackage.cluster_selection import SelectedCluster, RelevanceEvaluator, cluster_retrieval, context_expansion, context_expansion_generator, print_candidates
from mypackage.llm import LLMSession
from mypackage.storage import DatabaseSession, RealTimeResults, ExperimentManager

console = Console()
message_sender = None
_console_width = None

#===============================================================================================================

def retrieval_stage(sess, query: Query, exp_manager: ExperimentManager, *, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 1 of the pipeline
    '''
    message_sender("info", "Retrieving documents...")
    times['elastic'] = time.time()

    if not server_args.test_mode:
        res = query.execute(sess, size=args.num_documents)
    else:
        res = exp_manager.get_docs(None, sess)

    times['elastic'] = time.time() - times['elastic']
    message_sender("time", {'elastic': times['elastic']})
    message_sender("docs", [doc.id for doc in res])

    return res

#===============================================================================================================

def encode_query(query: Query, db: DatabaseSession, *, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 2 of the pipeline
    '''
    message_sender('info', "Encoding query...\n")

    times['query_encode'] = time.time()

    #Find out what model was used for this experiment
    params = db.get_experiment_params()
    model_name = params['sentence_model']

    sentence_model = SentenceTransformer(model_name, device='cpu')
    query.encode(sentence_model)
    times['query_encode'] = time.time() - times['query_encode']

    message_sender('time', {'query_encode': times['query_encode']})

#===============================================================================================================

def retrieve_clusters(sess, db, returned_docs, query, *, keep_cluster: int, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 3 of the pipeline
    '''
    message_sender('info', "Extracting relevant information...\n")

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

def calculate_cross_scores(query, selected_clusters: list[SelectedCluster], *, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 4 of the pipeline
    '''
    evaluator = RelevanceEvaluator(query, 'cross-encoder/ms-marco-MiniLM-L12-v2')

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

def expand_context(selected_clusters: list[SelectedCluster], *, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 5 of the pipeline
    '''
    cross_scores = []

    message_sender("info", "Expanding relevant information...\n")

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

        #Remove some of the very bad candidates, before context expansion even begins.
        #This saves time, because it's unlikely that very low candidates will manage to improve enough
        focused_cluster.candidates = [c for c in focused_cluster.candidates if c.score > args.cand_filter]

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
                'new_score': float(cluster.cross_score)
            }
            for i, cluster in enumerate(selected_clusters)
        ])

#===============================================================================================================

def summarization_stage(query: Query, selected_clusters: list[SelectedCluster], stop_dict, summary_num: int, *, args: Arguments = None, base_path: str = "..", times: defaultdict, server_args):
    '''
    Stage 6 of the pipeline
    '''
    #The text to be summarized
    unit = SummaryUnit(selected_clusters, sorting_method=args.csm)
    
    if args.print:
        res = unit.pretty_print(show_added_context=True, show_chain_indices=True, return_text=True, console_width=_console_width)
        message_sender("ansi_text", res)
        #res = rich_console_text(unit.text)
        #message_sender("ansi_text", res)

    if args.summ:
        is_first_fragment = True

        #Only measure for the first summary
        measure_response_time = (times['summary_response_time'] == 0)

        message_sender("info", "[white]Summarizing...[/white]" + (f" ({summary_num+1}/{args.num_summaries})" if args.num_summaries > 1 else ""))
        llm = LLMSession.create(server_args.llm_backend, "meta-llama-3.1-8b-instruct", api_host=server_args.llm_host)

        summarizer = Summarizer(query, llm=llm)
        if measure_response_time: times['summary_response_time'] = time.time()

        #Generate the fragments
        for fragment, citation in summarizer.summarize(unit, stop_dict, cache_prompt=True):
            
            if is_first_fragment:
                temp_generation_time = time.time()
                if measure_response_time:
                    times['summary_response_time'] = time.time() - times['summary_response_time']
                    message_sender('time', {'summary_response_time': times['summary_response_time']})
                message_sender("fragment", fragment)
                is_first_fragment = False
            else:
                if citation is not None:
                    message_sender("fragment_with_citation", {'fragment': fragment, 'citation': citation})
                else:
                    message_sender("fragment", fragment)

        temp_generation_time = time.time() - temp_generation_time
        times['generation_time'] += temp_generation_time
        message_sender('time', {'generation_time': times['generation_time']})

    return unit

#===============================================================================================================

def pipeline(
        query_data: dict,
        stop_dict,
        *,
        args: Arguments = None,
        server_args,
        base_path: str = ".",
        socket: SocketIO,
        console_width: int,
        store_as: str | None
    ):

    global message_sender, _console_width

    if args is None:
        args = Arguments()

    #Create a sender function that is bound to specific namespace. Saves time
    message_sender = lambda event, data: socket.emit(event, data, namespace="/query")

    if not args.summ and args.num_summaries > 1:
        args.num_summaries = 1
        message_sender("warn", "Because summarization is disabled, number of summaries was set to 1 (single empty summary)")

    _console_width = console_width #Match the client's console width
    times = defaultdict(float)
    kwargs = {'args': args, 'base_path': base_path, 'times': times, 'server_args': server_args}

    #Session initialization
    #--------------------------------------------------------------------------
    sess = Session(args.index, base_path=base_path, use="cache" if server_args.test_mode else "client", cache_dir="../cache")
    query  = Query.from_data(query_data)
    exp_manager = ExperimentManager("../common/experiments.json")
    db = DatabaseSession.init_db(server_args.db, exp_manager.db_name(sess.index_name), args.experiment)

    #Pipeline stages
    #--------------------------------------------------------------------------
    returned_docs = retrieval_stage(sess, query, exp_manager, **kwargs)
    encode_query(query, db, **kwargs)
    selected_clusters = retrieve_clusters(sess, db, returned_docs, query, keep_cluster=args.c, **kwargs)
    evaluator = calculate_cross_scores(query, selected_clusters, **kwargs)

    #Make a copy of the selected clusters that has an independent state
    #After we expand the context of our clusters, this original state stays the same
    original_selected_clusters = [sc.copy_selected_cluster() for sc in selected_clusters]
    #original_selected_clusters = None

    expand_context(selected_clusters, **kwargs)

    #Summarization
    summaries = []
    temp = kwargs['args'].print
    for i in range(args.num_summaries):
        summaries.append(summarization_stage(query, selected_clusters, stop_dict, i, **kwargs))
        message_sender("summary_end", None)
        kwargs['args'].print = False #Do not resend the input to the client
    kwargs['args'].print = temp

    #Terminate
    #--------------------------------------------------------------------------
    message_sender("end", {'status': 0})
    db.close()

    #Store results (for future evaluation)
    if store_as is not None:
        RealTimeResults.store_results(f"../eval/realtime_results/{store_as}", sess, query, evaluator, args.to_namespace(), returned_docs, original_selected_clusters, selected_clusters, summaries, times)