from .classes import RelevanceEvaluator
from ..elastic import Document
from ..cluster_selection import SelectedCluster, SummaryCandidate, context_expansion_generator, print_candidates, context_expansion, panel_print

from rich.console import Console
from rich.rule import Rule
import numpy as np
import time

from typing import overload, Literal, Union
from collections import defaultdict
from itertools import accumulate

from matplotlib import pyplot as plt
console = Console()

#===============================================================================================================

def single_document_cross_score(doc: Document, evaluator: RelevanceEvaluator, *, verbose: bool = False) -> float:
    '''
    Identifies all the chains of a specific document that are relevant to a query and sums up their scores.

    Arguments
    ---
    selected_clusters: list[SelectedCluster]
        The list of selected clusters that came specifically from the document ```doc```
    '''
    if verbose:
        console.print(Rule(f"Document {doc.id}"))

    #Finding the best chains of the document, regardless of cluster
    t = time.time()
    scores = evaluator.predict(doc.chains)
    positive_chains = [c for c in doc.chains if scores[c.index] > 0]
    positive_scores = [scores[c.index] for c in positive_chains]

    if verbose:
        panel_print([
            f"[green]Evaluation time:[/green] [cyan]{round(time.time() - t, 3):.3f}s[/cyan]",
            f"[green]Total score:[/green] [cyan]{np.round(np.sum(scores), 3):.3f}[/cyan]",
            f"[green]Positive only:[/green] [cyan]{np.round(np.sum(positive_scores), 3):.3f}[/cyan]"
        ],
        title="Results on the initial document")

    if len(positive_chains) > 0:

        #We create a fake SelectedCluster with only positie chains, so that we can apply context expansion on it
        #This helps us find a theoretical maximum score for this document
        #...while also taking the constraints into consideration (e.g. candidate thresholds)
        fake_selected_cluster = SelectedCluster(None, None, candidates=[SummaryCandidate(c, scores[c.index], evaluator=evaluator) for c in positive_chains])

        #Apply context expansion
        t = time.time()
        if verbose:
            for text in context_expansion_generator(fake_selected_cluster):
                print(text, end="")
        else:
            context_expansion(fake_selected_cluster)

        #Keep candidates that are above a threshold
        fake_selected_cluster.filter_and_merge_candidates()

        if verbose:
            print_candidates(fake_selected_cluster, title=f"Filtered candidates")
            panel_print([
                f"[green]Expansion time:[/green] [cyan]{round(time.time() - t, 3):.3f}s[/cyan]",
                f"[green]New score:[/green] [cyan]{fake_selected_cluster.cross_score:.3f}[/cyan]",
                #f"[green]Final result:[/green] [cyan]{round(real_score/fake_selected_cluster.cross_score, 3)}[/cyan]"
            ],
            title="Results after context expansion")

        #for c in retrieved_chains:
            #console.print(f"[cyan][{c.index}][/cyan] [#FF64DC]({round(scores[c.index], 3):.3f})[/#FF64DC]: [green]{c.text}[/green]\n")

        
        return fake_selected_cluster.cross_score
    
    else:
        return 0

#===============================================================================================================

@overload
def document_cross_score(docs: list[Document], selected_clusters: list[SelectedCluster], evaluator: RelevanceEvaluator, *, keep_all_docs: bool = False,  verbose: bool = False, vector: Literal[False]) -> float: ...

@overload
def document_cross_score(docs: list[Document], selected_clusters: list[SelectedCluster], evaluator: RelevanceEvaluator, *, keep_all_docs: bool = False,  verbose: bool = False, vector: Literal[True]) -> list[float]: ...

#@overload
#def document_cross_score(docs: list[Document], selected_clusters: list[SelectedCluster], evaluator: RelevanceEvaluator, *, verbose: bool = False, vector: bool = ...) -> Union[float, list[float]]: ...

def document_cross_score(docs: list[Document], selected_clusters: list[SelectedCluster], evaluator: RelevanceEvaluator, *, keep_all_docs: bool = False, verbose: bool = False, vector: bool = False) -> Union[float, list[float]]:
    if not keep_all_docs:
        #Only keep the documents that have clusters that passed the threshold
        docs = [doc for doc in docs if doc.id in [sc.doc.id for sc in selected_clusters]]

    #Group clusters by document
    #We want to sum up their scores to get a document score
    clusters_per_doc = defaultdict(list[SelectedCluster])
    for focused_cluster in selected_clusters:
        clusters_per_doc[focused_cluster.cluster.doc].append(focused_cluster)

    #Calculate the objective scores for the docs
    doc_scores = []
    for doc in docs:
        doc_scores.append(single_document_cross_score(doc, evaluator, verbose=verbose))

    if keep_all_docs:
        real_scores = [0.0 if len(clusters_per_doc[doc]) == 0 else sum(sc.cross_score for sc in clusters_per_doc[doc]) for doc in docs]
        return round(sum(real_scores)/sum(doc_scores), 3) if not vector else list(zip(real_scores, doc_scores))
    else:
        #The score of what we retrieved from the clusters
        real_scores = [sum(sc.cross_score for sc in clusters_per_doc[doc]) for doc in docs]

        scores = [1.0 if ds == 0 else round(rs/ds, 3) for rs, ds in zip(real_scores, doc_scores)]
        return scores if vector else round(sum(scores)/len(scores), 3)
    
#===============================================================================================================

def document_cross_score_at_k(scores: list[float]):
    fig, ax = plt.subplots()

    temp1 = list(accumulate(x for x,_ in scores))
    temp2 = list(accumulate(x for _,x in scores))
    temp = [t1/t2 for t1,t2 in zip(temp1, temp2)]

    ax.plot(range(1, len(scores)+1), temp)
    ax.set_xticks(range(1, len(scores)+1), labels=range(1, len(scores)+1))
    #ax.set_yticks(np.arange(0, 1.5, 0.05))
    plt.show(block=True)
