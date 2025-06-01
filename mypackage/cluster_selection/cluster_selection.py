
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

def print_candidates(focused_cluster: SelectedCluster, *, print_action: bool = False, current_state_only: bool = False):

    panel_lines = []

    for i, c in enumerate(focused_cluster.candidates):
        line_text = f"[green]{i:02}[/green]. "
        line_text_list = []

        col = "green" if c.expandable else "red"

        for num, state in enumerate([c.context] if current_state_only else c.history):
            #Yes I know this is goofy
            big_chain_in_column = False
            for c1 in focused_cluster.candidates:
                if len(c1.context if current_state_only else c1.history[num]) > 1:
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

    #Overall cluster score\
    if current_state_only:
        panel_lines.append(f"Cluster score: [cyan]{focused_cluster.cross_score:.3f}[/cyan]")
    else:
        cluster_scores = [
            f"[cyan]{focused_cluster.historic_cross_score(i):.3f}[/cyan]" for i in range(len(focused_cluster.candidates[0].history))
        ]
        panel_lines.append(f"Cluster score: " + " [red]->[/red] ".join(cluster_scores))

    panel_print(panel_lines, title=f"For cluster {focused_cluster.id}", expand=False)

#===============================================================================================================

def context_expansion(cluster: SelectedCluster):

    timestamp = 0

    while True:
    #for _ in range(2):
        expanded = False #Stop if nobody expands, 

        seen_chains = set()
        #This tells us which candidate caused a certain chain to be forbidden
        candidate_index_to_position: dict[int, int] = {}
        marked_for_deletion: list[bool] = []

        #Evaluate the different contexts
        #Candidates are in order of relevance
        for pos, candidate in enumerate(cluster.candidates):
        #----------------------------------------------------------------------------------------------------------
            #console.print(Rule())
            #console.print(candidate.index_range)
            #console.print(candidate.context.timestamp)
            marked_for_deletion.append(False)

            if not candidate.expandable:
                for c in candidate.context.chains:
                    seen_chains.add(c.index)
                    candidate_index_to_position[c.index] = pos
                continue

            #If I myself am forbidden, I just have to kill myself. It's that simple
            if candidate.chain.index in seen_chains:
                marked_for_deletion[-1] = True
                continue

            #Solidify the currently selected state if it's -1
            #otherwise the addition of context will change our state
            if candidate.selected_state == -1:
                candidate.selected_state = len(candidate.history) - 1

            #Refresh the timestamp of the current state
            #Unsure if I'll have to make a copy instead
            candidate.context.timestamp = timestamp
            
            if len(candidate.context.actions) == 0 or candidate.context.actions[-1].startswith("bidirectional"):
                branch_point = candidate.selected_state #We need to keep this, because we want to limit ourselves to the current timestamp
                candidate.add_left_context(timestamp=timestamp)
                candidate.add_right_context(branch_from=branch_point, timestamp=timestamp)
                candidate.add_bidirectional_context(branch_from=branch_point, timestamp=timestamp)
            elif candidate.context.actions[-1].startswith("left"):
                candidate.add_left_context(timestamp=timestamp)
            elif candidate.context.actions[-1].startswith("right"):
                candidate.add_right_context(timestamp=timestamp)
            
            #candidate.optimize(stop_expansion=True)
            candidate.optimize(stop_expansion=True, timestamp=timestamp)

            #Check if the new state is forbidden
            while True:
                forbidden_chains = [c for c in candidate.context.chains if c.index in seen_chains]

                if len(forbidden_chains) == 0:
                    break
                if len(forbidden_chains) == 1:
                    #Because someone better than us restricted us, that some has actually already been scored
                    #Let's see if that candidate that restricted us is still better
                    other_candidate = cluster.candidates[candidate_index_to_position[forbidden_chains[0].index]]
                    if candidate.score > other_candidate.score:
                        marked_for_deletion[candidate_index_to_position[forbidden_chains[0].index]] = True
                        candidate_index_to_position[forbidden_chains[0].index] = pos
                        break
                    else:
                        #If this extra chain was forbidden, then I need to delete this state from the history
                        #I then need to find the immediately next best state, and check if that is forbidden too
                        #(Non-terminating)
                        candidate.history.pop(candidate.selected_state)
                        #candidate.optimize()
                        candidate.optimize(timestamp=timestamp)

                elif len(forbidden_chains) == 2:
                    #This is a more serious case
                    #We need to beat both of the candidates that restrict us
                    #Only then is it beneficial for the current candidate to exist
                    other1 = cluster.candidates[candidate_index_to_position[forbidden_chains[0].index]]
                    other2 = cluster.candidates[candidate_index_to_position[forbidden_chains[1].index]]

                    if candidate.score > other1.score and candidate.score > other2.score:
                        marked_for_deletion[candidate_index_to_position[forbidden_chains[0].index]] = True
                        marked_for_deletion[candidate_index_to_position[forbidden_chains[1].index]] = True
                        candidate_index_to_position[forbidden_chains[0].index]
                        candidate_index_to_position[forbidden_chains[0].index] = pos
                        candidate_index_to_position[forbidden_chains[1].index] = pos
                        break
                    else:
                        #The only state we can return to is the initial state
                        #And we can no longer expand, since we are restricted from both sides
                        candidate.selected_state = [i for i, s in enumerate(candidate.history) if s.timestamp == timestamp][0]
                        #candidate.selected_state = 0
                        candidate.expandable = False
                        break

            #candidate.clear_history()
            candidate.clear_timestamp(timestamp-1)
            #Right now, forbidden_chains has all the chains that are forbidden in the current state
            #I either forbade them already myself, or someone else forbade them
            #The remaining chains, I have to forbid myself
            for c in candidate.context.chains:
                if c.index not in forbidden_chains:
                    seen_chains.add(c.index)
                    candidate_index_to_position[c.index] = pos

            expanded |= candidate.expandable

        #console.print(candidate_index_to_position)

        #Delete those that were marked for deletion
        cluster.candidates = [c for i,c in enumerate(cluster.candidates) if not marked_for_deletion[i]]

        #Clear history
        for candidate in cluster.candidates:
            candidate.clear_history()

        cluster.remove_duplicate_candidates().rerank_candidates()
        print_candidates(cluster, print_action=True, current_state_only=True)

        if not expanded:
            break

        timestamp += 1