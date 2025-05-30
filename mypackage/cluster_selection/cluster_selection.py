
from ..elastic import Session, ElasticDocument
from ..query import Query
from .classes import SelectedCluster
from ..storage import load_pickles
from ..helper import panel_print

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from operator import methodcaller

from rich.console import Console
from rich.rule import Rule

console = Console()

#=====================================================================================================
    
def cluster_retrieval(sess: Session, docs: list[ElasticDocument], query: Query, method: str = "thres") -> list[SelectedCluster]:
    #Load the clusters corresponding to the retrieved documents
    pkl_list = load_pickles(sess, "../experiments/pubmed-index/pickles/default", docs = docs)

    #Extract all the clusters from all the retrieved documents, into one container
    #Keep track which document each cluster came from
    #Ignore outlier clusters
    clusters = []
    doc_labels = []

    for doc_number, pkl in enumerate(pkl_list):
        for cluster in pkl.clustering:
            if cluster.label > -1:
                clusters.append(cluster)
                doc_labels.append(doc_number)

    #visualize_clustering(clusters, doc_labels, show=True)

    #Find the similarity to each cluster centroid
    #Select best clusters
    #----------------------------------------------------------------------------------------------------------
    sim = cosine_similarity([cluster.vector for cluster in clusters], query.vector.reshape((1,-1)))
    sorted_clusters = [[np.round(x[0], decimals=3), x[1], x[2]] for x in sorted(zip(map(methodcaller("__getitem__", 0), sim), clusters, doc_labels), reverse=True)]

    selected_clusters = []
    selected_clusters: list[SelectedCluster]

    if method == "topk":
        #Mark top k clusters
        k = 7
        for i in range(k):
            sorted_clusters[i][2] = 11
            #print(sorted_clusters[i][0])
            console.print((sorted_clusters[i][1].doc.id, sorted_clusters[i][0]))
            selected_clusters.append(SelectedCluster(sorted_clusters[i][1], sorted_clusters[i][0]))
    elif method == "thres":
        thres = 0.5
        for cluster in sorted_clusters:
            if cluster[0] > thres:
                cluster[2] = 11
                selected_clusters.append(SelectedCluster(cluster[1], cluster[0]))
            else:
                break

    return selected_clusters

#===============================================================================================================

def print_candidates(focused_cluster: SelectedCluster, *, print_action: bool = False):

    panel_lines = []

    for i, c in enumerate(focused_cluster.candidates):
        line_text = f"[green]{i:02}[/green]. "
        line_text_list = []

        col = "green" if c.expandable else "red"

        for num, state in enumerate(c.history):
            #Yes I know this is goofy
            big_chain_in_column = False
            for c1 in focused_cluster.candidates:
                if len(c1.history[num]) > 1:
                    big_chain_in_column = True
                    break

            if not big_chain_in_column:
                temp = f"[{col}]{state.chains[0].index:03}[/{col}]"
            else:   
                temp = f"[{col}]{state.chains[0].index:03}[/{col}]".rjust(19).ljust(23 if c.expandable else 19) if len(state) == 1 else f"[{col}]{state.id}[/{col}]"

            history_text = f"Chain {temp}" if len(state) == 1 else f"Chains {temp}"
            history_text += f" with score " + f"[cyan]{state.score:.3f}[/cyan]".rjust(20)
            history_text += f" ({' -> '.join(state.actions)})".ljust(19) if print_action else ""
            line_text_list.append(history_text)

        line_text += " [red]->[/red] ".join(line_text_list)
        panel_lines.append(line_text)

    panel_lines.append(Rule())

    #Overall cluster score
    cluster_scores = [
        f"[cyan]{focused_cluster.historic_cross_score(i):.3f}[/cyan]" for i in range(len(focused_cluster.candidates[0].history))
    ]

    panel_lines.append(f"Cluster score: " + " [red]->[/red] ".join(cluster_scores))

    panel_print(panel_lines, title=f"For cluster {focused_cluster.id}", expand=False)

#===============================================================================================================

def context_expansion(cluster: SelectedCluster):

    while True:
        expanded = False #Stop if nobody expands, 
        #Evaluate the different contexts
        for candidate in cluster.candidates:
            if not candidate.expandable:
                continue

            #Solidify the currently selected state if it's -1
            #otherwise the addition of context will change our state
            candidate.selected_state = len(candidate.history) - 1

            '''
            candidate.add_left_context()
            candidate.add_right_context(branch_from=0)
            candidate.add_bidirectional_context(branch_from=0)
            '''
            
            #Identify candidate's direction based on where it moved last
            if len(candidate.context.actions) == 0 or candidate.context.actions[-1].startswith("bidirectional"):
                candidate.add_left_context()
                candidate.add_right_context(branch_from=0)
                candidate.add_bidirectional_context(branch_from=0)
            elif candidate.context.actions[-1].startswith("left"):
                candidate.add_left_context()
            elif candidate.context.actions[-1].startswith("right"):
                candidate.add_right_context()
            
            
            candidate.optimize(stop_expansion=True)
            candidate.clear_history()

            expanded |= candidate.expandable

        cluster.remove_duplicate_candidates().rerank_candidates()
        print_candidates(cluster, print_action=True)

        if not expanded:
            break