from .classes import RelevanceEvaluator
from ..elastic import Document
from ..cluster_selection import SelectedCluster, SummaryCandidate, context_expansion_generator, print_candidates, context_expansion, panel_print
from ..sentence import SentenceChain
from ..clustering import ChainCluster,  ChainClustering

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

def single_document_cross_score(doc: Document, evaluator: RelevanceEvaluator, *, verbose: bool = False, cand_filter: float = 0) -> tuple[float, float]:
    '''
    Identifies all the chains of a specific document that are relevant to a query and sums up their scores.

    Returns
    ---
    doc_score: float
        The document's cross-encoder score
    eval_t: float
        Time it took to evaluate the document
    '''
    if verbose:
        console.print(Rule(f"Document {doc.id}"))

    t = time.time()
    eval_t = time.time()

    #Create an artificial cluster, with all the sentences of the document as single-element chains
    sentences = [SentenceChain(s, chain_index=i) for i,s in enumerate(doc.sentences)]
    parent_cluster = ChainCluster(sentences, 0) 
    parent_cluster.clustering_context = ChainClustering(sentences, [0]*len(sentences), {0: parent_cluster})
    for s in sentences:
        s.parent_cluster = parent_cluster

    #Evaluate chains and pick out the positive ones
    scores = evaluator.predict(sentences)
    positive_sentences = [c for c in sentences if scores[c.index] > cand_filter]
    positive_scores = [scores[c.index] for c in positive_sentences]

    if verbose:
        panel_print([
            f"[green]Evaluation time:[/green] [cyan]{round(time.time() - t, 3):.3f}s[/cyan]",
            f"[green]Total score:[/green] [cyan]{np.round(np.sum(scores), 3):.3f}[/cyan]",
            f"[green]Positive only:[/green] [cyan]{np.round(np.sum(positive_scores), 3):.3f}[/cyan]"
        ],
        title="Results on the initial document")

    if len(positive_sentences) > 0:
        #We create an artificial SelectedCluster with only positive chains, so that we can apply context expansion on it
        #This helps us find a theoretical maximum score for this document
        #...while also taking the constraints into consideration (e.g. candidate thresholds)
        fake_selected_cluster = SelectedCluster(None, None, candidates=[SummaryCandidate(c, scores[c.index], evaluator=evaluator) for c in positive_sentences])

        #Apply context expansion
        #----------------------------------------------------------------------------
        t = time.time()
        if verbose:
            for text in context_expansion_generator(fake_selected_cluster):
                print(text, end="")
        else:
            context_expansion(fake_selected_cluster)

        #Keep candidates that are above a threshold
        fake_selected_cluster.filter_and_merge_candidates()
        t = time.time() - t
        eval_t = time.time() - eval_t
        #----------------------------------------------------------------------------

        if verbose:
            print_candidates(fake_selected_cluster, title=f"Filtered candidates")
            panel_print([
                f"[green]Expansion time:[/green] [cyan]{round(t, 3):.3f}s[/cyan]",
                f"[green]New score:[/green] [cyan]{fake_selected_cluster.cross_score:.3f}[/cyan]",
                #f"[green]Final result:[/green] [cyan]{round(real_score/fake_selected_cluster.cross_score, 3)}[/cyan]"
            ],
            title="Results after context expansion")
        
        return fake_selected_cluster.cross_score, eval_t
    else:
        return 0, time.time() - eval_t

#===============================================================================================================

#@overload
#def document_cross_score(docs: list[Document], selected_clusters: list[SelectedCluster], evaluator: RelevanceEvaluator, *, keep_all_docs: bool = False,  verbose: bool = False, vector: Literal[False]) -> float: ...

#@overload
#def document_cross_score(docs: list[Document], selected_clusters: list[SelectedCluster], evaluator: RelevanceEvaluator, *, keep_all_docs: bool = False,  verbose: bool = False, vector: Literal[True]) -> list[float]: ...

##@overload
##def document_cross_score(docs: list[Document], selected_clusters: list[SelectedCluster], evaluator: RelevanceEvaluator, *, verbose: bool = False, vector: bool = ...) -> Union[float, list[float]]: ...

def document_cross_score(docs: list[Document], selected_clusters: list[SelectedCluster], evaluator: RelevanceEvaluator, *, keep_all_docs: bool = True, verbose: bool = False, vector: bool = False, cand_filter: float = 0) -> Union[float, list[float]]:
    '''
    Calculates the value we managed to extract from a set of documents relative to a query by only examining the relevant clusters.
    We fist calculate a theoretical maximum score for each document.
    This is done by considering each sentence as its own chain, then including all of the document's sentneces
    into one artificial cluster. We then apply the same context expansion algorithm on the cluster to maximize its score

    Arguments
    ---
    docs: list[Document]
        The list of documents to evaluate
    selected_clusters: list[SelectedCluster]
        The clusters to evaluate
    evaluator: RelevanceEvaluator
        The evaluator
    keep_all_docs: bool
        If ```True```, then all retrieved documents are considered as part of the evaluation. In this case,
        even if we didn't retrieve a cluster for some document, we still count the score we missed, counting the retrieved score as 0.
        Defaults to ```False```
    verbose: bool
        Verbose
    vector: bool
        Return vector. Defaults to ```False```

    Returns
    ---
    cluster_selection_score: float|list[float]
        The final score
    doc_times: list[float]
        The times it took to evaluate each documents
    '''
    if not keep_all_docs:
        #Only keep the documents that have clusters that passed the threshold
        docs = [doc for doc in docs if doc.id in [sc.doc.id for sc in selected_clusters]]

    #Group clusters by document
    #We want to sum up their scores to get a recalled document score
    clusters_per_doc = defaultdict(list[SelectedCluster])
    for focused_cluster in selected_clusters:
        clusters_per_doc[focused_cluster.cluster.doc].append(focused_cluster)

    #Calculate the objective scores for the docs
    doc_scores = []
    doc_times = []
    for doc in docs:
        local_score, doc_t = single_document_cross_score(doc, evaluator, verbose=verbose, cand_filter=cand_filter)
        doc_scores.append(local_score)

        #We want to compare the time it took to evaluate all of the doc's sentences separately
        #...with the time it took to only look at the relevant clusters
        #So let's gather the doc times
        doc_times.append(doc_t)

    if keep_all_docs:
        real_scores = [0.0 if len(clusters_per_doc[doc]) == 0 else sum(sc.cross_score for sc in clusters_per_doc[doc]) for doc in docs]
        return (round(sum(real_scores)/sum(doc_scores), 3) if not vector else list(zip(real_scores, doc_scores)), doc_times)
    else:
        #The score of what we retrieved from the clusters
        real_scores = [sum(sc.cross_score for sc in clusters_per_doc[doc]) for doc in docs]

        scores = [1.0 if ds == 0 else round(rs/ds, 3) for rs, ds in zip(real_scores, doc_scores)]
        return (scores if vector else round(sum(scores)/len(scores), 3), doc_times)
    
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
