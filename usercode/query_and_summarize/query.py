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
from dataclasses import dataclass, field, fields
from typing import Literal, get_origin, get_args, Any
import json

from rich.pretty import Pretty
from rich.console import Console
from rich.live import Live
from rich.rule import Rule
from rich.padding import Padding
from rich.tree import Tree

console = Console()

@dataclass
class Arguments():
    c: int = field(default=0, metadata={"help": "The cluster number"})
    s: Literal["topk", "thres"] = field(default="thres", metadata={"help": "Best cluster selection method"})
    print: bool = field(default=False, metadata={"help": "Print the cluster's text"})
    stats: bool = field(default=False, metadata={"help": "Print cluster stats"})
    summ: bool = field(default=True, metadata={"help": "Summarize"})
    cet: float = field(default=0.01, metadata={"help": "Context expansion threshold"})
    csm: str = field(default="flat_relevance", metadata={"help": "Candidate sorting method"})


@dataclass
class Message():
    type: str
    contents: Any

    def to_json(self, string=False) -> dict:
        data = {'type': self.type, 'contents': self.contents}
        if string:
            return json.dumps(data)
        return data

    def to_sse(self):
        return f"data: {self.to_json(string=True)}\n\n".encode("utf-8")

#===============================================================================================================

def query(query_str: str, *, args: Arguments = None, base_path="."):
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

    panel_print([f"[green]{i:02}.[/green] Cluster [green]{cluster.id}[/green] with score [cyan]{cluster.sim:.3f}[/cyan]" for i, cluster in enumerate(selected_clusters)], title="Retrieved clusters based on cosine similarity")

    #Print cluster stats
    if args.stats:
        panel_print([Pretty(cluster_stats(cluster)) for cluster in selected_clusters], title="Cluster Stats")

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

        with Live(panel_print(return_panel=True), refresh_per_second=10) as live:
            for fragment in summarizer.summarize(unit):
                if is_first_fragment:
                    times['summary_response_time'] = time.time() - times['summary_response_time']
                    is_first_fragment = False
                yield Message("fragment", fragment)
                live.update(panel_print(unit.summary, return_panel=True))

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

    yield Message("end", 1)

#===============================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    for f in fields(Arguments):
        if f.type is bool:
            parser.add_argument(f"--{f.name}", action="store_true", dest=f.name, default=f.default, help=f.metadata['help'])
            if f.default == True:
                parser.add_argument(f"--no-{f.name}", action="store_false", dest=f.name, help=f.metadata['help'])
        elif get_origin(f.type) is Literal:
            parser.add_argument(f"-{f.name}", action="store", type=str, default=f.default, help=f.metadata['help'], choices=list(get_args(f.type)))
        else:
            parser.add_argument(f"-{f.name}", action="store", type=f.type, default=f.default, help=f.metadata['help'])

    args = parser.parse_args()

    console.print(Pretty(args))
    for msg in query("Query", args=args, base_path=".."):
        pass

    