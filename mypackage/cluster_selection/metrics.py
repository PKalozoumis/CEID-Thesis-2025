from .classes import RelevanceEvaluator
from ..elastic import Document
from ..cluster_selection import SelectedCluster, SummaryCandidate, context_expansion_generator, print_candidates, context_expansion

from rich.console import Console
from rich.rule import Rule
import numpy as np
import time

console = Console()

def single_document_cross_score(doc: Document, selected_clusters: list[SelectedCluster], evaluator: RelevanceEvaluator):
    '''
    Identifies all the chains of a specific document that are relevant to a query and sums up their scores.

    Arguments
    ---
    selected_clusters: list[SelectedCluster]
        The list of selected clusters that came specifically from the document ```doc```
    '''
    #test
    sc = selected_clusters[0]

    t = time.time()
    scores = evaluator.predict(doc.chains)
    console.print(f"Evaluation time: {round(time.time() - t, 3):.3f}s")
    console.print(f"Total score: {np.round(np.sum(scores), 3):.3f}")

    positive_chains = [c for c in doc.chains if scores[c.index] > 0]
    positive_scores = [scores[c.index] for c in positive_chains]
    console.print(f"Positive only: {np.round(np.sum(positive_scores), 3):.3f}")

    #console.print(Rule("Document Chains"))
    #for c in positive_chains:
        #console.print(f"[cyan][{c.index}][/cyan] [#FF64DC]({round(scores[c.index], 3):.3f})[/#FF64DC]: [green]{c.text}[/green]\n")

    retrieved_chains = sc.context_chains(flat=True)

    console.print(Rule("Context Expansion on document chains"))


    fake_selected_cluster = SelectedCluster(None, None, candidates=[SummaryCandidate(c, scores[c.index], evaluator=evaluator) for c in positive_chains])

    context_expansion(fake_selected_cluster)
    #for text in context_expansion_generator(fake_selected_cluster):
        #print(text, end="")

    #Keep candidates that are above a threshold
    fake_selected_cluster.filter_candidates()

    #Send final result of expansion to client
    #res = print_candidates(fake_selected_cluster, title=f"Filtered candidates")

    #for c in retrieved_chains:
        #console.print(f"[cyan][{c.index}][/cyan] [#FF64DC]({round(scores[c.index], 3):.3f})[/#FF64DC]: [green]{c.text}[/green]\n")

    {
        'precision': None,
        'recall': None
    }