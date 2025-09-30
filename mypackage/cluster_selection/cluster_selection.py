
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from operator import methodcaller
import inspect
import textwrap

from rich.console import Console

from .classes import SelectedCluster
from .helper import print_candidates
if TYPE_CHECKING:
    from ..storage import DatabaseSession
    from ..elastic import Session, ElasticDocument
    from ..query import Query

console = Console()

#=====================================================================================================
    
def cluster_retrieval(sess: Session, db: DatabaseSession, docs: list[ElasticDocument], query: Query, method: str = "thres") -> list[SelectedCluster]:
    '''
    Retrieve the best clusters from the database that correspond to the retrieved documents

    Arguments
    ---
    sess: Session
        The Elasticsearch session
    db: DatabaseSession
        The preprocessing database session
    docs: list[ElasticDocument]
        The relevant documents
    query: Query
        The query
    method: str
        Method to select the best clusters from the relevant docs.
        Options are ```topk``` (keep top-k relevant clusters) and ```thres``` (keep those above threshold).
        Defaults to ```thres```

    Returns
    ---
    res: list[SelectedCluster]
        The selected clusters
    '''
    #Load the clusters corresponding to the retrieved documents
    processed_docs = db.load(sess, docs)

    #Extract all the clusters from all the retrieved documents, into one container
    #Keep track which document each cluster came from
    #Ignore outlier clusters
    clusters = []
    doc_labels = []

    for doc_number, proc_doc in enumerate(processed_docs):
        for cluster in proc_doc.clustering:
            if cluster.label > -1:
                clusters.append(cluster)
                doc_labels.append(doc_number)

    #Find the similarity to each cluster centroid
    #Sort in decreasing order of similarity to the query
    #Select best clusters
    #----------------------------------------------------------------------------------------------------------
    sim = cosine_similarity([cluster.vector for cluster in clusters], query.vector.reshape((1,-1)))
    sorted_clusters = [[np.round(x[0], decimals=3), x[1], x[2]] for x in sorted(zip(map(methodcaller("__getitem__", 0), sim), clusters, doc_labels), reverse=True)]

    selected_clusters = []
    selected_clusters: list[SelectedCluster]

    if method == "topk":
        k = 7
        for i in range(k):
            sorted_clusters[i][2] = 11 #marks the cluster as selected. debug only
            console.print((sorted_clusters[i][1].doc.id, sorted_clusters[i][0]))
            selected_clusters.append(SelectedCluster(sorted_clusters[i][1], sorted_clusters[i][0]))
    elif method == "thres":
        thres = 0.55
        for cluster in sorted_clusters:
            if cluster[0] > thres:
                cluster[2] = 11 #marks the cluster as selected. debug only
                selected_clusters.append(SelectedCluster(cluster[1], cluster[0]))
            else:
                break

    return selected_clusters

#===============================================================================================================

def context_expansion_generator(cluster: SelectedCluster, *, threshold:  float = 0.01):
    '''
    spaghetti code
    '''

    timestamp = 0
    prev_round_forbids = None #Key is index of forbidden chain, value is who forbade it

    while True:
    #for _ in range(1):
        expanded = False #Stop if nobody expands, 

        seen_chains = set()
        #This tells us which candidate caused a certain chain to be forbidden
        current_round_forbids: dict[int, int] = {} #Key is index of forbidden chain, value is who forbade it
        marked_for_deletion: list[bool] = [False]*len(cluster.candidates)

        #Evaluate the different contexts
        #Candidates are in order of relevance
        for pos, candidate in enumerate(cluster.candidates):
        #----------------------------------------------------------------------------------------------------------
            #console.print(Rule())
            #console.print(candidate.index_range)
            #console.print(candidate.context.timestamp)
            
            if marked_for_deletion[pos]:
                continue

            if not candidate.expandable:
                for c in candidate.context.chains:
                    seen_chains.add(c.index)
                    current_round_forbids[c.index] = pos
                continue

            #If I myself am forbidden, I just have to kill myself. It's that simple
            if candidate.chain.index in seen_chains:
                marked_for_deletion[pos] = True
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
            
            candidate.optimize(stop_expansion=True, timestamp=timestamp, threshold=threshold)

            #Check if the new state is forbidden
            while True:
                forbidden_chains = [c for c in candidate.context.chains if c.index in seen_chains]

                if len(forbidden_chains) == 0:
                    break
                if len(forbidden_chains) == 1:
                    #Because someone better than us restricted us, that someone has actually already been scored
                    #Let's see if that candidate that restricted us is still better
                    other_candidate = cluster.candidates[current_round_forbids[forbidden_chains[0].index]]
                    if candidate.score > other_candidate.score:
                        marked_for_deletion[current_round_forbids[forbidden_chains[0].index]] = True
                        current_round_forbids[forbidden_chains[0].index] = pos
                        break
                    else:
                        #If this extra chain was forbidden, then I need to delete this state from the history
                        #I then need to find the immediately next best state, and check if that is forbidden too
                        #(Non-terminating)
                        initial_state_for_this_timestamp = [i for i, s in enumerate(candidate.history) if s.timestamp == timestamp][0]
                        candidate.history.pop(candidate.selected_state)
                        candidate.selected_state = None
                        candidate.optimize(timestamp=timestamp, threshold=threshold)
                        if candidate.selected_state == initial_state_for_this_timestamp:
                            candidate.expandable = False

                elif len(forbidden_chains) == 2:
                    #This is a more serious case
                    #We need to beat both of the candidates that restrict us
                    #Only then is it beneficial for the current candidate to exist
                    other1 = cluster.candidates[current_round_forbids[forbidden_chains[0].index]]
                    other2 = cluster.candidates[current_round_forbids[forbidden_chains[1].index]]

                    if candidate.score > other1.score and candidate.score > other2.score:
                        marked_for_deletion[current_round_forbids[forbidden_chains[0].index]] = True
                        marked_for_deletion[current_round_forbids[forbidden_chains[1].index]] = True
                        current_round_forbids[forbidden_chains[0].index] = pos
                        current_round_forbids[forbidden_chains[1].index] = pos
                        break
                    else:
                        #The only state we can return to is the initial state
                        #And we can no longer expand, since we are restricted from both sides
                        candidate.selected_state = [i for i, s in enumerate(candidate.history) if s.timestamp == timestamp][0]
                        #candidate.selected_state = 0
                        candidate.expandable = False
                        break

            candidate.clear_timestamp(timestamp-1)

            #Right now, forbidden_chains has all the chains that are forbidden in the current state
            #I either forbade them already myself, or someone else forbade them
            #The remaining chains, I have to forbid myself
            for c in candidate.context.chains:
                if c.index not in seen_chains:
                    seen_chains.add(c.index)
                    current_round_forbids[c.index] = pos

            #After checking if my current chains are forbidden, I now have a final set of chains (that I have also forbidden)
            #...I now need to check if anyone below me acquired these chains in the previous round
            #I need to force them to choose something else (from their previous round choices)
            #(which are also their only choices, they have not reached the current round yet)
            if prev_round_forbids is not None:
                for ch in candidate.context.chains:
                    if pos_to_forbid := prev_round_forbids.get(ch.index, None):
                        if pos_to_forbid > pos:
                            pos_to_forbid: int
                            bad = cluster.candidates[pos_to_forbid]

                            #We need to force him to choose something else
                            if bad.optimize(constraints=list(current_round_forbids.keys()), threshold=threshold) is None:
                                marked_for_deletion[pos_to_forbid] = True

                            #print(f"I AM {candidate.context.id} AND I NEED TO FORBID CHAIN {ch.index} AT POSITION {pos_to_forbid}")

            expanded |= candidate.expandable

        #console.print(candidate_index_to_position)

        #Delete those that were marked for deletion
        cluster.candidates = [c for i,c in enumerate(cluster.candidates) if not marked_for_deletion[i]]

        cluster.remove_duplicate_candidates().rerank_candidates()
        yield print_candidates(cluster, print_action=True, current_state_only=True, return_text=True)

        prev_round_forbids = {}
        for pos, candidate in enumerate(cluster.candidates):
            for ch in candidate.context.chains:
                prev_round_forbids[ch.index] = pos

        if not expanded:
            break

        timestamp += 1

    #print_candidates(cluster, print_action=True, current_state_only=False)

    #Clear history
    for candidate in cluster.candidates:
        candidate.clear_history()

#Copy the exact same function, but make it a non-generator
#The generator version is meant to return progress messages
src = inspect.getsource(context_expansion_generator)
src = textwrap.dedent(src)
src = src.replace("def context_expansion_generator", "def context_expansion")
src = src.replace("yield ", "")

exec(src, globals())