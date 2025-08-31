from __future__ import annotations

from dataclasses import dataclass, field
import pickle
import os
import numpy as np
from sentence_transformers import CrossEncoder
from itertools import chain

from rich.rule import Rule
from rich.padding import Padding
from rich.pretty import Pretty
from rich.console import Console

from ..sentence import SentenceChain
from ..query import Query
from ..clustering import ChainCluster, ChainClustering
from ..helper import panel_print
from ..sentence import SentenceLike
from ..elastic import Document
from .helper import print_candidates, print_candidate_states

import time

console = Console()

#==========================================================================================================

@dataclass
class RelevanceEvaluator():
    '''
    Contains the query, as well as the model used to evaluate a sentence's relevance with it
    '''
    query: Query
    model_name: str
    model: CrossEncoder = field(default=None)

    def __post_init__(self):
        if self.model is None:
            self.model = CrossEncoder(self.model_name)

    def predict(self, sentences: list[SentenceLike], *, join=False) -> float | list[float]:
        '''
        Uses the cross-encoder to evaluate each sentence or chain

        Arguments
        ---
        sentences: list[SentenceLike]
            The list of sentences or chains to evaluate
        join: bool
            If set to ```True```, it evaluates all sentences in the input list as one input sequence. Defaults to ```False```,
            meaining that each sentence is evaluated separately
        '''
        if join:
            return self.model.predict([(self.query.text, "".join([s.text for s in sentences]))])[0]
        else:
            return self.model.predict([(self.query.text, c.text) for c in sentences])

#==========================================================================================================

class SummaryCandidate():
    '''
    Represents a central chain on which we apply context expansion.
    It is the input and output of the context expansion process.
    It is a list of one or more consecutive chains whose text may be used as input to the summarization model.
    The relevance score of the full span is calculated with a cross-encoder.
    '''
    chain: SentenceChain #The central chain around which we build the context
    selected_state: int #The optimal state from the history
    history: list[State] #States to show how the addition of more context affects the score
    evaluator: RelevanceEvaluator
    #Whether this candidate should be considered for further context expansions
    #Candidate Filtering can disable this candidate if no improvement is seen
    expandable: bool

    #---------------------------------------------------------------------------------------------------

    @dataclass
    class State():
        '''
        A specific context expansion state. Contains all the chains of the context, along with its score
        '''
        chains: list[SentenceChain]
        score: float
        actions: list[str] = field(default_factory=list)
        timestamp: int = field(default=0)
        improvement_score: float = field(default=0)

        @classmethod
        def from_state(cls, state: SummaryCandidate.State, action: str = None) -> SummaryCandidate.State:
            '''
            Constructs a new state by copying an existing state.

            Arguments
            ---
            state: State
                The state to copy
            action: str, optional
                An optional action to add to the list of existing actions

            Returns
            ---
            state: State
                The new state
            '''
            return cls(
                chains = state.chains[:],
                score = state.score,
                actions = state.actions + ([action] if action is not None else []),
                timestamp = state.timestamp,
                improvement_score = state.improvement_score
            )
        
        @property
        def id(self) -> str:
            return f"{self.chains[0].index:03}-{self.chains[-1].index:03}" if len(self.chains) > 1 else f"{self.chains[0].index:03}"
        
        @property
        def index_range(self) -> range:
            '''
            The range is inclusive
            '''
            return range(self.chains[0].index, self.chains[-1].index + 1)
        
        @property
        def first_index(self) -> int:
            return self.chains[0].index
        
        @property
        def last_index(self) -> int:
            return self.chains[-1].index
        
        @property
        def first_sentence_index(self) -> int:
            return self.chains[0].first_index
        
        @property
        def last_sentence_index(self) -> int:
            return self.chains[-1].last_index
        
        @property
        def text(self):
            return "".join([c.text for c in self.chains])
        
        def __len__(self) -> int:
            return len(self.chains)
        
        def __eq__(self, other: SummaryCandidate.State) -> bool:
            return self.id == other.id and self.actions == other.actions
        
        #----------------------------------------------------------------------
        
        def data(self) -> dict:
            return {
                'chains': [c.index for c in self.chains],
                'score': self.score,
                'actions': self.actions,
                'timestamp': self.timestamp,
                'improvement_score': self.improvement_score
            }
        
        @classmethod
        def from_data(cls, data, all_chains: list[SentenceChain]):
            return cls(
                [all_chains[i] for i in data['chains']],
                data['score'],
                data['actions'],
                data['timestamp'],
                data['improvement_score']
            )
        
    #---------------------------------------------------------------------------------------------------

    def __init__(self, chain: SentenceChain, score: float, evaluator: RelevanceEvaluator = None):
        self.chain = chain
        self.selected_state = -1 #Latest
        self.history = [self.State([chain], score)]
        self.evaluator = evaluator
        self.expandable = True

    @property
    def context(self) -> State:
        return self.history[self.selected_state]
    
    @property
    def id(self) -> str:
        return f"{self.chain.doc.id}_{self.context.id}"
    
    @property
    def citation(self) -> str:
        return f"<{self.chain.doc.id}_{self.first_sentence_index}-{self.last_sentence_index}>"
    
    @property
    def score(self):
        return self.context.score
    
    @score.setter
    def score(self, value: float):
        self.context.score = value

    @property
    def text(self):
        return " ".join([c.text for c in self.context.chains])
    
    def pretty_text(self, *, show_added_context = False, show_chain_indices = False, show_chain_sizes = False):

        thing1 = lambda c: f"[cyan][{c.index}" + (f" ({len(c)} sentences)" if show_chain_sizes else "") + "]: [/cyan]"
        thing2 = lambda c: c.text if c.index == self.chain.index else f"[green]{c.text}[/green]"

        return "".join([
            (thing1(c) if show_chain_indices else "") +
            (thing2(c) if show_added_context else c.text)
            for c in self.context.chains
        ])
    
    @property
    def index_range(self):
        '''
        The range is inclusive
        '''
        return self.context.index_range
    
    @property
    def first_index(self):
        return self.context.first_index
    
    @property
    def last_index(self):
        return self.context.last_index
    
    @property
    def first_sentence_index(self):
        return self.context.first_sentence_index
    
    @property
    def last_sentence_index(self):
        return self.context.last_sentence_index

    #---------------------------------------------------------------------------------------------------

    def optimize(self, *, stop_expansion: bool = False, timestamp: int|None = None, constraints: list[int]|None = None, threshold: float = 0.01) -> int:
        '''
        Sets the selected state to the optimal state. Returns the new selected index

        Arguments
        ---
        stop_expansion: bool
            When set to ```True```, if the optimal state is the same as the old state, the
            candidate is marked as non-expandable, meaning that no improvement occurs from expansion

        timestamp: int, optional
            Only optimize on the states that have specific timestamp

        constraints: list[int], optional
            Prevent the optimization process from selecting states that contain the chains in this list

        threshold: float
            Threshold for how much the score needs to improve relative to the size of the new context. Defaults to ```0.01```
        '''
        if timestamp is not None:
            temp = [i for i, s in enumerate(self.history) if s.timestamp == timestamp]
        else:
            temp = range(len(self.history))

        if constraints is not None:
            #Only keep states that don't have chains in the constraints list
            temp =  [i for i in temp if not any(it.index in constraints for it in self.history[i].chains)]

        #This sets the current state as the bare minimum
        #If no one else is above the threshold, we default to the current state
        if self.selected_state is not None:
            self.context.improvement_score = threshold

        if len(temp) > 0:
            new_state = max(temp, key=lambda i: self.history[i].improvement_score)
            #print(f"I am chain {self.chain.index}, optimal_state={new_state}, current_state={self.selected_state}")

            if stop_expansion and self.selected_state == new_state: #Optimal state did not change
                self.expandable = False

            self.selected_state = new_state
            return new_state
        else:
            self.expandable = False
            return None

    #---------------------------------------------------------------------------------------------------

    def add_right_context(self, n: int = 1, *, branch_from: int|None = None, timestamp: int = 0):
        '''
        Adds extra context (chains) to the right of the current state

        Arguments
        ---
        n: int
            The number of extra chains to add. Defaults to ```1```

        branch_from: int | None
            The index of the state from ```history``` from which to expand. Defaults to ```None```,
            meaning the currently selected state (denoted by ```selected_state```). Setting to ```-1```
            expands from the latest state in the history
        '''
        if n <= 0 or not self.expandable:
            return
        
        if branch_from is None:
            branch_from = self.selected_state
            if branch_from == -1:
                branch_from = len(self.history) - 1

        self.history.append(self.State.from_state(self.history[branch_from], f"right {n}"))
        self.history[-1].timestamp = timestamp
        self.history[-1].chains.extend(self.history[-1].chains[-1].next(n, force_list=True))
        self.history[-1].score = self.evaluator.predict(self.history[-1].chains, join=True) #Evaluate the new context

        #added_length = len(self.history[-1].text.split()) - len(self.history[branch_from].text.split())
        added_length = len(self.history[-1].text.split())
        if added_length > 0:
            self.history[-1].improvement_score = round((self.history[-1].score - self.history[branch_from].score)/added_length, 3)

    #---------------------------------------------------------------------------------------------------

    def add_left_context(self, n: int = 1, *, branch_from: int|None = None, timestamp: int = 0):
        '''
        Adds extra context (chains) to the left of the current state

        Arguments
        ---
        n: int
            The number of extra chains to add. Defaults to ```1```

        branch_from: int | None
            The index of the state from ```history``` from which to expand. Defaults to ```None```,
            meaning the currently selected state (denoted by ```selected_state```). Setting to ```-1```
            expands from the latest state in the history
        '''
        if n <= 0 or not self.expandable:
            return
        
        if branch_from is None:
            branch_from = self.selected_state
            if branch_from == -1:
                branch_from = len(self.history) - 1

        self.history.append(self.State.from_state(self.history[branch_from], f"left {n}"))
        self.history[-1].timestamp = timestamp
        self.history[-1].chains = self.history[-1].chains[0].prev(n, force_list=True) + self.history[-1].chains
        self.history[-1].score = self.evaluator.predict(self.history[-1].chains, join=True) #Evaluate the new context

        #added_length = len(self.history[-1].text.split()) - len(self.history[branch_from].text.split())
        added_length = len(self.history[-1].text.split())
        if added_length > 0:
            self.history[-1].improvement_score = round((self.history[-1].score - self.history[branch_from].score)/added_length, 3)

    #---------------------------------------------------------------------------------------------------

    def add_bidirectional_context(self, n: int = 1, *, branch_from: int|None = None, timestamp: int = 0):
        '''
        Adds extra context (chains) to both directions of the current state

        Arguments
        ---
        n: int
            The number of extra chains to add. Defaults to ```1```

        branch_from: int | None
            The index of the state from ```history``` from which to expand. Defaults to ```None```,
            meaning the currently selected state (denoted by ```selected_state```). Setting to ```-1```
            expands from the latest state in the history
        '''
        if n <= 0 or not self.expandable:
            return
        
        if branch_from is None:
            branch_from = self.selected_state
            if branch_from == -1:
                branch_from = len(self.history) - 1

        self.history.append(self.State.from_state(self.history[branch_from], f"bidirectional {n}"))
        self.history[-1].timestamp = timestamp
        self.history[-1].chains.extend(self.history[-1].chains[-1].next(n, force_list=True)) #Forward
        self.history[-1].chains = self.history[-1].chains[0].prev(n, force_list=True) + self.history[-1].chains #backward
        self.history[-1].score = self.evaluator.predict(self.history[-1].chains, join=True) #Evaluate the new context

        #added_length = len(self.history[-1].text.split()) - len(self.history[branch_from].text.split())
        added_length = len(self.history[-1].text.split())
        if added_length > 0:
            self.history[-1].improvement_score = round((self.history[-1].score - self.history[branch_from].score)/added_length, 3)

    #---------------------------------------------------------------------------------------------------

    def print_history(self):
        text = []
        for state in self.history:
            text.append(("[red]-->[/red] " if self.context == state else "") + f"[cyan]{state.first_index}-{state.last_index}[/cyan]: Score = {state.score:.3f}, Timestamp = {state.timestamp}, Actions = [green]{state.actions}[/green]")
            text.append(Rule())

        text = text[:-1]
        panel_print(text, title=f"For candidate {self.chain.index:03}")

    #---------------------------------------------------------------------------------------------------

    def clear_history(self, exceptions: list[int] = []):
        '''
        Clears the entire history, except for the currently selected state and the indices in ```exceptions``` 
        '''
        had_latest_state = False
        if self.selected_state == -1:
            had_latest_state = True
            self.selected_state = len(self.history) - 1

        preserved = set(exceptions + [self.selected_state])
        self.history = [state for i, state in enumerate(self.history) if i in preserved]

        if had_latest_state:
            self.selected_state = -1
        else:
            # Recalculate current_index since history may have shrunk
            old_to_new = {i: new_i for new_i, i in enumerate(sorted(preserved))}
            self.selected_state = old_to_new.get(self.selected_state, 0)

    #-------------------------------------------------------------------------------------------------------------------

    def clear_timestamp(self, timestamp: int):
        '''
        Clears the history entries that have the specified ```timestamp```
        '''
        if self.context.timestamp == timestamp:
            raise Exception("You cannot clear entries with the same timestamp as the current timestamp")
        had_latest_state = False
        if self.selected_state == -1:
            had_latest_state = True
            self.selected_state = len(self.history) - 1

        old_history = self.history
        self.history = [state for state in self.history if state.timestamp != timestamp]

        if had_latest_state:
            self.selected_state = -1
        else:
            # Adjust selected_state if items were removed before it
            removed_before = sum(1 for i in range(len(old_history)) 
                                if old_history[i].timestamp == timestamp and i < self.selected_state)
            self.selected_state -= removed_before
            self.selected_state = max(0, min(self.selected_state, len(self.history) - 1))
    
    #---------------------------------------------------------------------------------------------------

    def __str__(self) -> str:
        return f"SummaryCandidate(range=[{self.context.chains[0].index}, {self.context.chains[-1].index}], score={self.score:.3f})"
    
    def __repr__(self) -> str:
        return f"SummaryCandidate(range=[{self.context.chains[0].index}, {self.context.chains[-1].index}], score={self.score:.3f})"
    
    #---------------------------------------------------------------------------------------------------

    def data(self) -> dict:
        return {
            'chain_index': self.chain.index,
            'selected_state': self.selected_state,
            'history': [state.data() for state in self.history],
            'expandable': self.expandable
        }
    
    @classmethod
    def from_data(cls, data, all_chains: list[SentenceChain], evaluator: RelevanceEvaluator) -> SummaryCandidate:
        temp = cls(all_chains[data['chain_index']], 0)
        temp.selected_state = data['selected_state']
        temp.history = [cls.State.from_data(d, all_chains) for d in data['history']]
        temp.evaluator = evaluator
        temp.expandable = data['expandable']
        return temp


#==========================================================================================================

@dataclass
class SelectedCluster():
    '''
    Represents a cluster that is semanticall close to a given query.
    It contains a list of chains, along with their cross-encoder similarity scores
    to the query. The overall cluster is further classified based on these partial scores
    '''

    cluster: ChainCluster
    sim: float #Similarity to query
    candidates: list[SummaryCandidate] = field(default=None, kw_only=True) #Looks confusing, but it's essentially the chains of the cluster, sorted by score
    evaluator: RelevanceEvaluator = field(default=None, kw_only=True)

    #Temporary. For debugging only. Please never use
    #---------------------------------------------------------------------------
    def store_scores(self, base_path:str) -> dict:
        res = {}
        for candidate in self.candidates:
            res[candidate.chain.index] = candidate.score

        with open(os.path.join(base_path, f"{self.id}.pkl"), "wb") as f:
            pickle.dump(res, f)

    def load_scores(self, base_path:str) -> dict:
        with open(os.path.join(base_path, f"{self.id}.pkl"), "rb") as f:
            data = pickle.load(f)

        temp = [(data[chain.index], chain) for chain in self.cluster.chains]

        self.candidates = [SummaryCandidate(chain, score, self.evaluator) for score, chain in sorted(temp, reverse=True)]

    #---------------------------------------------------------------------------

    @property
    def cross_score(self) -> float:
        '''
        A relevance score for the entire cluster, by summing up the individual cross-encoder scores of the chains
        '''
        if self.candidates is None:
            return None

        return np.round(sum(self.scores()), decimals=3)
    
    #---------------------------------------------------------------------------
    
    @property
    def id(self) -> str:
        return self.cluster.id if self.cluster is not None else None
    
    @property
    def text(self) -> str:
        #Sort by start
        temp = sorted(self.candidates, key=lambda x: x.first_index, reverse=False)
        return "\n\n".join([t.text for t in temp])
    
    @property
    def doc(self) -> Document:
        return self.cluster.doc
    
    @property
    def pretty_text(self) -> str:
        #Sort by start
        temp = sorted(self.candidates, key=lambda x: x.first_index, reverse=False)
        return "\n\n".join([t.text for t in temp])
    
    @property
    def clustering_context(self) -> ChainClustering:
        return self.cluster.clustering_context
    
    #---------------------------------------------------------------------------
    
    def evaluate_chains(self) -> SelectedCluster:
        '''
        Calculates the cross-encoder similarity score between the query and each chain in the cluster.
        After execution, each chain is transformed into a ```SummaryCandidate```, all of which are stored in the ```candidates``` list
        '''
        scores = self.evaluator.predict(self.cluster.chains)
        self.candidates = [SummaryCandidate(chain, score, self.evaluator) for score, chain in sorted(zip(scores, self.cluster.chains), reverse=True)]

        return self
    
    #---------------------------------------------------------------------------
    
    def remove_duplicate_candidates(self) -> SelectedCluster:
        seen = set()
        keep = []
        for candidate in self.candidates:
            temp = tuple(candidate.index_range)
            if temp not in seen:
                keep.append(candidate)
                seen.add(temp)
        
        self.candidates = keep
        return self
    
    #---------------------------------------------------------------------------

    def rescore_candidates(self) -> SelectedCluster:
        '''
        After changing chains of some candidates, you want to recalculate their scores.
        During context expansion, this happens automatically, but if you manually modify a chain, you also have to rescore
        '''
        for candidate in self.candidates:
            candidate.history[candidate.selected_state].score = candidate.evaluator.predict(candidate.history[candidate.selected_state].chains, join=True) #Evaluate the new context
        return self

    #---------------------------------------------------------------------------
    
    def rerank_candidates(self) -> SelectedCluster:
        '''
        Sort candidates in decreasing order of score
        '''
        self.candidates = sorted(self.candidates, key=lambda x: (x.score, 6666 - x.chain.index), reverse=True)
        return self
    
    #---------------------------------------------------------------------------

    def filter_and_merge_candidates(self, negative_threshold: float = -2, good_candidate_threshold: float = 3, max_bridge_size: int = 1) -> SelectedCluster:
        '''
        Only keeps candidates that have a cross-score above the threshold
        '''
        filtered_candidates = [c for c in self.candidates if c.score > negative_threshold]
        good_candidates = [c for c in filtered_candidates if c.score >= good_candidate_threshold]
        bad_candidates = [c for c in filtered_candidates if c.score < good_candidate_threshold]

        #For every good candidate, try to find a bad candidate that is within bridging distance
        for gc in good_candidates:
            for bc in bad_candidates:
                forward_dista = bc.first_index - gc.last_index 
                backward_dista = gc.first_index - bc.last_index

                #Candidates can be bridged (forward)
                if 0 <= forward_dista <= max_bridge_size + 1:
                    #console.print(f"Forward bridging: candidate {gc.id} can be bridged with {bc.id}")
                    #Try out the following scenarios:
                    gc.expandable = True
                    for i in range(len(bc.context)):
                        gc.add_right_context(forward_dista + i)
                    #print_candidate_states(gc)

                #Candidates can be bridged (backward)
                elif 0 <= backward_dista <= max_bridge_size + 1:
                    #console.print(f"Backward bridging: candidate {gc.id} can be bridged with {bc.id}")
                    #Try out the following scenarios:
                    gc.expandable = True
                    for i in range(len(bc.context)):
                        gc.add_left_context(backward_dista + i)
                    #print_candidate_states(gc)
                    
                #Select the best scenario
                gc.optimize()
                gc.clear_history()
                gc.expandable = False

        #After this, reject all the bad candidates
        self.candidates = good_candidates

        #Merge remaining candidates
        if len(good_candidates) > 0:
            return self._merge_candidates()
        return self
    #---------------------------------------------------------------------------
    
    def _merge_candidates(self) -> SelectedCluster:
        '''
        Merges neighboring chains into one

        Merges candidates that contain overlapping chains. A merge only happens between candidates whose scores
        have the same sign (both positive or negative)

        NOTE: Maybe this won't be necessary, since the context expansion no longer generates overlapping chains
        '''
        
        self.candidates = sorted(self.candidates, key=lambda x: x.first_index, reverse=False)

        prev = self.candidates[0]
        keep = []
        for candidate in self.candidates[1:]:
            #There is overlap
            #I don't think this can happen anymore...
            if candidate.index_range.start in prev.index_range:
                print("Overlap detected")
                #How many chains do we need to add?
                extra_chains = candidate.index_range.stop - prev.index_range.stop
                prev.context.chains += candidate.context.chains[len(candidate.context.chains)-extra_chains:]
            #Neighbors
            elif candidate.index_range.start == prev.index_range.stop:
                prev.context.chains += candidate.context.chains
            else:
                keep.append(prev)
                prev = candidate

        keep.append(prev)

        self.candidates = keep
        self.rescore_candidates().rerank_candidates()

        return self
    
    #---------------------------------------------------------------------------

    def central_chains(self) -> list[SentenceChain]:
        '''
        List of the releavance-sorted chains in descending order
        '''
        return [c.chain for c in self.candidates]
        
    #---------------------------------------------------------------------------
        
    def context_chains(self, flat: bool = False) -> list[SentenceChain] | list[list[SentenceChain]]:
        '''
        List of the releavance-sorted context chains in descending order
        '''
        if not flat:
            return [c.context.chains for c in self.candidates]
        else:
            return list(chain.from_iterable(c.context.chains for c in self.candidates))
    
    #---------------------------------------------------------------------------
    
    def scores(self) -> list[float]:
        '''
        List of the chain scores in descending order
        '''
        return [c.score for c in self.candidates]
    
    #---------------------------------------------------------------------------

    def historic_cross_score(self, i: int) -> float:
        '''
        Return the entire cluster's score at a specific point in time.
        This quietly assumes that all contexts move at the same pace (all histories have same length)
        '''
        if self.candidates is None:
            return None

        return np.round(sum([c.history[i].score for c in self.candidates]), decimals=3)
        
    #---------------------------------------------------------------------------

    def print(self):
        group = []
        for i, chain in enumerate(self.cluster.chains):
            group.append(Pretty(f"{i:02}. Chain {chain}"))
            group.append(Rule())
            group.append(Padding(chain.text, pad=(0,0,2,0)))

        panel_print(group)

    #---------------------------------------------------------------------------

    def __len__(self):
        return len(self.cluster)
        
    def __iter__(self):
        return iter(self.candidates)
    
    #---------------------------------------------------------------------------

    def data(self) -> dict:
        return {
            #The actual cluster will be loaded from the preprocessing results. We assign at a later time
            'cluster_label': self.cluster.label,
            #For validity check during loading. We must not pass the wrong ChainClustering
            'doc_id': self.cluster.doc.id,
            'sim': self.sim,
            'candidates': [sc.data() for sc in self.candidates]
        }
    
    @classmethod
    def from_data(cls, data: dict, clustering: ChainClustering, evaluator: RelevanceEvaluator):
        #This requires that you have loaded the preprocessing results first

        #Check if the clustering is ok
        if clustering.doc.id != data['doc_id']:
            raise Exception(f"ChainClustering object of document {clustering.doc.id} does not match document {data['doc_id']} from data")

        all_chains = clustering.chains

        return cls(
            clustering.clusters[data['cluster_label']],
            data['sim'],
            candidates = [SummaryCandidate.from_data(sc_data, all_chains, evaluator) for sc_data in data['candidates']],
            evaluator=evaluator
        )