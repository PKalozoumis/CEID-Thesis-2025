from ..sentence import SentenceChain
from ..query import Query
from ..clustering import ChainCluster, ChainClustering
from ..helper import panel_print
from ..sentence import SentenceLike

from dataclasses import dataclass, field
import pickle
import os
import numpy as np
from sentence_transformers import CrossEncoder

from rich.rule import Rule
from rich.padding import Padding
from rich.pretty import Pretty

#==========================================================================================================

@dataclass
class RelevanceEvaluator():
    '''
    Contains the query, as well as the model used to evaluate relevance with it
    '''
    query: Query
    model: CrossEncoder

    def predict(self, sentences: list[SentenceLike], *, join=False) -> float | list[float]:
        if join:
            return self.model.predict([(self.query.text, "".join([s.text for s in sentences]))])[0]
        else:
            return self.model.predict([(self.query.text, c.text) for c in sentences])

#==========================================================================================================

class SummaryCandidate():
    '''
    Represents a list of one or more consecutive chains whose text may be used as input to the summarization model,
    depending on their relevance to the query. The relevance score of the full span is calculated with a cross-encoder
    '''
    chain: SentenceChain #The central chain around which we build the context
    selected_state: int #The optimal state from the history
    history: list['State'] #States to show how the addition of more context affects the score
    evaluator: RelevanceEvaluator
    #Whether this candidate should be considered for further context expansions
    #Candidate Filtering can disable this candidate if no improvement is seen
    expandable: bool

    @dataclass
    class State():
        chains: list[SentenceChain]
        score: float
        actions: list[str] = field(default_factory=list)

        @classmethod
        def from_state(cls, state: 'SummaryCandidate.State', action: str) -> 'SummaryCandidate.State':
            return cls(
                chains = state.chains[:],
                score = state.score,
                actions = state.actions + [action]
            )
        
        def __len__(self) -> int:
            return len(self.chains)
        
        @property
        def id(self) -> str:
            return f"{self.chains[0].index:03}-{self.chains[-1].index:03}"

    def __init__(self, chain: SentenceChain, score: float, evaluator: RelevanceEvaluator = None):
        self.chain = chain
        self.selected_state = -1 #Latest
        self.history = [self.State([chain], score)]
        self.evaluator = evaluator
        self.expandable = True

    def __str__(self) -> str:
        return f"SummaryCandidate(range=[{self.context.chains[0].index}, {self.context.chains[-1].index}], score={self.score:.3f})"

    @property
    def context(self) -> State:
        return self.history[self.selected_state]
    
    @property
    def score(self):
        return self.context.score
    
    @score.setter
    def score(self, value: float):
        self.context.score = value

    @property
    def text(self):
        return "".join([c.text for c in self.context.chains])
    
    @property
    def index_range(self):
        return range(self.context.chains[0].index, self.context.chains[-1].index + 1)
    
    def optimize(self, *, stop_expansion: bool = False) -> int:
        '''
        Sets the selected state to the optimal state. Returns the new selected index

        Arguments
        ---
        stop_expansion: bool
            When set to ```True```, if the optimal state is the same as the old state, the
            candidate is marked as non-expandable, meaning that no improvement occurs from expansion
        '''
        new_state = max(range(len(self.history)), key=lambda i: self.history[i].score)
        #print(f"I am chain {self.chain.index}, optimal_state={new_state}, current_state={self.selected_state}")

        if stop_expansion and self.selected_state == new_state: #Optimal state did not change
            self.expandable = False

        self.selected_state = new_state
        return new_state
    
    #-------------------------------------------------------------------------------------------------------------------

    def add_right_context(self, n: int = 1, *, branch_from: int|None = None):
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
        if not self.expandable:
            return
        
        if branch_from is None:
            branch_from = self.selected_state

        self.history.append(self.State.from_state(self.history[branch_from], f"right {n}"))
        self.history[-1].chains.extend(self.history[-1].chains[-1].next(n, force_list=True))
        self.history[-1].score = self.evaluator.predict(self.history[-1].chains, join=True) #Evaluate the new context

    #-------------------------------------------------------------------------------------------------------------------

    def add_left_context(self, n: int = 1, *, branch_from: int|None = None):
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
        if not self.expandable:
            return
        
        if branch_from is None:
            branch_from = self.selected_state

        self.history.append(self.State.from_state(self.history[branch_from], f"left {n}"))
        self.history[-1].chains = self.history[-1].chains[0].prev(n, force_list=True) + self.history[-1].chains
        self.history[-1].score = self.evaluator.predict(self.history[-1].chains, join=True) #Evaluate the new context

    #-------------------------------------------------------------------------------------------------------------------

    def add_bidirectional_context(self, n: int = 1, *, branch_from: int|None = None):
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
        if not self.expandable:
            return
        
        if branch_from is None:
            branch_from = self.selected_state

        self.history.append(self.State.from_state(self.history[branch_from], f"bidirectional {n}"))
        self.history[-1].chains.extend(self.history[-1].chains[-1].next(n, force_list=True)) #Forward
        self.history[-1].chains = self.history[-1].chains[0].prev(n, force_list=True) + self.history[-1].chains #backward
        self.history[-1].score = self.evaluator.predict(self.history[-1].chains, join=True) #Evaluate the new context

    #-------------------------------------------------------------------------------------------------------------------

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

#==========================================================================================================

@dataclass
class SelectedCluster():
    '''
    Represents a cluster that is semanticall close to a given query.
    It contains a list of chains, along with their cross-encoder similarity scores
    to the query. The overall cluster is further classified based on these partial score
    '''

    cluster: ChainCluster
    sim: float #Similarity to query
    candidates: list[SummaryCandidate] = field(default=None, kw_only=True) #Looks confusing, but it's essentially the chains of the cluster, sorted by score
    evaluator: RelevanceEvaluator = field(default=None, kw_only=True)

    #Temporary. For debugging only. Please never use or i will kill myself (real)
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
    
    def evaluate_chains(self) -> 'SelectedCluster':
        '''
        Calculates the cross-encoder similarity score between the query and each chain in the cluster.
        After execution, each chain is transformed into a ```SummaryCandidate```, all of which are stored in the ```candidates``` list
        '''
        scores = self.evaluator.predict(self.cluster.chains)
        self.candidates = [SummaryCandidate(chain, score, self.evaluator) for score, chain in sorted(zip(scores, self.cluster.chains), reverse=True)]

        return self
    
    #---------------------------------------------------------------------------
    
    def remove_duplicate_candidates(self) -> 'SelectedCluster':
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

    def refresh_candidate_scores(self) -> 'SelectedCluster':
        '''
        After changing chains of some candidates, you want to recalculate their scores.
        During context expansion, this happens automatically, but if you manually modify a chain, you also have to rescore
        '''
        for candidate in self.candidates:
            candidate.history[candidate.selected_state].score = candidate.evaluator.predict(candidate.history[candidate.selected_state].chains, join=True) #Evaluate the new context
        return self

    #---------------------------------------------------------------------------
    
    def rerank_candidates(self) -> 'SelectedCluster':
        self.candidates = sorted(self.candidates, key=lambda x: (x.score, 6666 - x.chain.index), reverse=True)
        return self

    #---------------------------------------------------------------------------
    
    def merge_candidates(self) -> 'SelectedCluster':
        self.candidates = sorted(self.candidates, key=lambda x: x.index_range.start, reverse=False)

        prev = self.candidates[0]
        keep = []
        for candidate in self.candidates[1:]:
            if prev.score*candidate.score >= 0:
                #There is overlap
                if candidate.index_range.start in prev.index_range:
                    #How many chains do we need to add?
                    extra_chains = candidate.index_range.stop - prev.index_range.stop
                    prev.context.chains += candidate.context.chains[len(candidate.context.chains)-extra_chains:]
                #Neighbors
                elif candidate.index_range.start == prev.index_range.stop:
                    prev.context.chains += candidate.context.chains
                else:
                    keep.append(prev)
                    prev = candidate
            else:
                keep.append(prev)
                prev = candidate

        keep.append(prev)

        self.candidates = keep
        self.refresh_candidate_scores().rerank_candidates()

        return self
    
    #---------------------------------------------------------------------------

    def central_chains(self) -> list[SentenceChain]:
        '''
        List of the releavance-sorted chains in descending order
        '''
        return [c.chain for c in self.candidates]
        
    #---------------------------------------------------------------------------
        
    def context_chains(self) -> list[list[SentenceChain]]:
        '''
        List of the releavance-sorted context chains in descending order
        '''
        return [c.context_chains for c in self.candidates]
    
    #---------------------------------------------------------------------------
    
    def scores(self) -> list[float]:
        '''
        List of the chain scores in descending order
        '''
        return [c.score for c in self.candidates]
    
    #---------------------------------------------------------------------------
    
    @property
    def cross_score(self) -> float:
        '''
        A relevance score for the entire cluster, by summing up the individual cross-encoder scores of the chains
        '''
        if self.candidates is None:
            return None

        return np.round(sum([score for score in self.scores()]), decimals=3)
    
    #---------------------------------------------------------------------------

    def historic_cross_score(self, i: int) -> float:
        '''
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

    @property
    def id(self) -> str:
        return self.cluster.id
    
    @property
    def text(self) -> str:
        #Sort by start
        temp = sorted(self.candidates, key=lambda x: x.index_range.start, reverse=False)
        if self.cross_score > 0:
            #We take all the candidates    
            return "\n\n".join([c.text for c in temp])
        else:
            #We only keep candidates of positive score
            return "\n\n".join([c.text for c in temp if c.score > 0])
    
    @property
    def clustering_context(self) -> ChainClustering:
        return self.cluster.clustering_context
    
    def __len__(self):
        return len(self.cluster)
        
    def __iter__(self):
        return iter(self.candidates)