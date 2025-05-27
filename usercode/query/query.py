import os
import sys
sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import Session, Query, ElasticDocument
from mypackage.storage import load_pickles
from rich.console import Console
from mypackage.clustering import visualize_clustering, ChainCluster, ChainClustering
from mypackage.clustering.metrics import cluster_stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from operator import methodcaller
from matplotlib import pyplot as plt
from mypackage.helper import panel_print
from sentence_transformers import CrossEncoder
from rich.rule import Rule
from rich.padding import Padding
from rich.pretty import Pretty
import numpy as np
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer, PegasusForConditionalGeneration
from mypackage.sentence import SentenceChain, SentenceLike
from rich.live import Live
from rich.panel import Panel
import argparse
from mypackage.summarization import evaluate_summary_relevance
import json
import copy
from itertools import chain
import pickle

from dataclasses import dataclass, field

console = Console()

#================================================================================================================

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

#================================================================================================================

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

#================================================================================================================

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
    
#================================================================================================================

#TODO
@dataclass
class SummarySegment():
    '''
    The input and output of the summarization system.
    Represents the part/paragraph of the summary referring to one specific document.
    Contains all the retrieved clusters that are relevant from the specific document, as well as general information
    about the document itself that should be included in the summary (e.g. author and main topic)
    '''
    clusters: list[SelectedCluster]
    doc: ElasticDocument
    #Could either be the paper's existing summary, or an LLM-generated text of the first paragraph or summary
    #...maybe it could even be pre-calculated?
    extra_info: str = field(default=None) 
    summary: str = field(default=None)
    citations: list[int] = field(default=None)

    def summarize(model):
        pass

#================================================================================================================

#TODO
class Summarizer():
    '''
    Contains all the models necessary for summarization. Also contains a connection to the LLM.
    Classifies the provided ```SummarySegments``` and summarizes them using the appropriate model.
    The final output is the summary
    '''
    sus: PegasusForConditionalGeneration #𐐘
    megasus: BigBirdPegasusForConditionalGeneration #ඞ


    def __init__(self):
        pass
    
#==============================================================================================================

def store_summary(cluster: SelectedCluster, summary: str, args):
    os.makedirs("generated_summaries", exist_ok=True)

    with open(f"generated_summaries/cluster_{cluster.id}.json", "w") as f:
        json.dump({
            'generated_with': "llm" if args.m == "llm" else "transformer",
            'summary': summary
        }, f)

#==============================================================================================================

def load_summary(cluster: SelectedCluster):
    if not os.path.exists(f"generated_summaries/cluster_{cluster.id}.json"):
        return None

    with open(f"generated_summaries/cluster_{cluster.id}.json", "r") as f:
        return json.load(f)['summary']

#===============================================================================================================

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

#================================================================================================================

def summarize(args, cluster: SelectedCluster):
    if args.cache:
        summary = load_summary(cluster)

    if not args.cache or summary is None:

        if args.m == "sus":
            #Let's try summarization of the entire cluster
            tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")
            summ_model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed")
            inputs = tokenizer(cluster.text, return_tensors='pt', truncation=True, max_length=4096)

            prediction = summ_model.generate(**inputs)
            prediction = tokenizer.batch_decode(prediction)

            panel_print(prediction)
            summary = prediction

        #----------------------------------------------------------------------------------------

        elif args.m == "llm":
            from mypackage.llm import summarize
            
            full_text = ""
            removed_json = False

            #Retrieve fragments of text from the llm and add them to the full text
            #------------------------------------------------------------------------------
            with Live(panel_print(return_panel=True), refresh_per_second=10) as live:
                for fragment in summarize(query.text, cluster.text):

                    full_text += fragment

                    #Clean up the json
                    #-------------------------------------------------
                    if fragment.endswith("}"):
                        full_text = full_text[:-2]

                    if not removed_json and full_text.startswith("{\"summary\": \""):
                        full_text = full_text[13:]
                        removed_json = True

                    #Once json is cleaned up, print
                    #-------------------------------------------------
                    else:
                        live.update(panel_print(full_text, return_panel=True))

            summary = full_text

        store_summary(selected_clusters[args.c], summary, args)

    #Evaluating the generated summary
    evaluate_summary_relevance(cross_model, summary, query.text)

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
            '''
            
            candidate.optimize(stop_expansion=True)
            candidate.clear_history()

            expanded |= candidate.expandable

        cluster.remove_duplicate_candidates().rerank_candidates()
        print_candidates(cluster, print_action=True)

        if not expanded:
            break

#===============================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", action="store", type=int, default=0, help="The cluster number")
    parser.add_argument("-m", action="store", type=str, default="sus", help="The model to use", choices=["sus", "llm"])
    parser.add_argument("-s", action="store", type=str, default="thres", help="Best cluster selection method", choices=["topk", "thres"])
    parser.add_argument("--print", action="store_true", default=False, help="Print the cluster's text")
    parser.add_argument("--stats", action="store_true", default=False, help="Print cluster stats")
    parser.add_argument("--cache", dest="cache", action="store_true", default=True, help="Try to load summary from cache")
    parser.add_argument("--no-cache", dest="cache", action="store_false", help="Try to load summary from cache")

    args = parser.parse_args()

    console.print(Pretty(args))

    #Retrieval stage
    #-----------------------------------------------------------------------------------------------------------------
    #sess = Session("pubmed-index", base_path="..")
    sess = Session("pubmed-index", use="cache", cache_dir="../cache")
    query = Query(0, "What are the primary behaviours and lifestyle factors that contribute to childhood obesity", source=["summary", "article"], text_path="article", cache_dir="cache")
    #res = query.execute(sess)

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
        ElasticDocument(sess, id=6415, text_path="article")
    ]

    #-----------------------------------------------------------------------------------------------------------------

    #Encode the query
    if not os.path.exists("query.npy"):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    else:
        model = None
    query.load_vector(model)

    #Retrieve clusters from docs
    selected_clusters = cluster_retrieval(sess, returned_docs, query)

    panel_print([f"Cluster [green]{cluster.id}[/green] with score [cyan]{cluster.sim:.3f}[/cyan]" for cluster in selected_clusters], title="Retrieved clusters based on cosine similarity")

    #Print cluster stats
    if args.stats:
        panel_print([Pretty(cluster_stats(cluster)) for cluster in selected_clusters], title="Cluster Stats")

    evaluator = RelevanceEvaluator(query, CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2'))

    #Calculate the cross-encoder scores
    #-----------------------------------------------------------------------------------------------------------------
    os.makedirs("cross_scores", exist_ok=True)
    for cluster in selected_clusters:
        cluster.evaluator = evaluator
        if not os.path.exists(f"cross_scores/{cluster.id}.pkl"):
            cluster.evaluate_chains()
            cluster.store_scores("cross_scores")
        else:
            cluster.load_scores("cross_scores")

    panel_print([f"Cluster {cluster.id} score: [cyan]{cluster.cross_score:.3f}[/cyan]" for cluster in selected_clusters], title="Cross-encoder scores of the selected clusters")

    if args.c != -1:
        selected_clusters = [selected_clusters[args.c]]

    #-----------------------------------------------------------------------------------------------------------------

    for focused_cluster in selected_clusters:
        focused_cluster: SelectedCluster

        #Print cluster chains
        if args.print:
            focused_cluster.print()

        #Let's evaluate chains
        print_candidates(focused_cluster)

        #Context Expansion
        #-----------------------------------------------------------------------------------------------------------------
        context_expansion(focused_cluster)
        focused_cluster.merge_candidates()
        print_candidates(focused_cluster)
        panel_print(focused_cluster.text, title=f"Text (size = {len(focused_cluster.text.split())})")

        #Summarization
        #-----------------------------------------------------------------------------------------------------------------
        #Create summary segments out of the selected clusters
        #For now we'll play with just one cluster
        seg = SummarySegment([focused_cluster], focused_cluster.cluster.doc)
        summarizer = Summarizer()