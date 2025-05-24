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
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
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

class SummaryCandidate():
    '''
    Represents a list of one or more consecutive chains whose text may be used as input to the summarization model,
    depending on their relevance to the query. The relevance score of the full span is calculated with a cross-encoder
    '''
    @dataclass
    class State():
        chains: list[SentenceChain]
        score: float

    chain: SentenceChain #The central chain around which we build the context
    main_context: State #The representative state that we want to use
    context_stages: list[tuple[State, str]] #Optional states to show how the addition of more context affects the score. Tuple of (state, action). Main not included
    model: CrossEncoder #Model used for evaluation

    def __init__(self, chain: SentenceChain, score: float, model: CrossEncoder = None):
        self.chain = chain
        self.main_context = self.State([chain], score)
        self.context_stages = None
        self.model = model

    @property
    def context_chains(self):
        return self.main_context.chains
    
    @property
    def score(self):
        return self.main_context.score
    
    @score.setter
    def score(self, value: float):
        self.main_context.score = value

    @property
    def text(self):
        return "".join([c.text for c in self.chains])
    
    @property
    def index_range(self):
        return range(self.chains[0].index, self.chains[-1].index + 1)
    
    def add_forward_context(self, n: int = 1):
        self.context_stages.append((copy.copy(self.main_context), f"forward {n}"))
        self.main_context.chains.extend(self.main_context.chains[-1].next(n, force_list=True))

    def add_backward_context(self, n: int = 1):
        self.context_stages.append((copy.copy(self.main_context), f"backward {n}"))
        self.main_context.chains = self.main_context.chains[0].prev(n, force_list=True) + self.main_context.chains

#================================================================================================================

@dataclass
class SelectedCluster():
    '''
    Represents a cluster that is semanticall close to a given query.
    It contains a list of chains, along with their cross-encoder similarity scores
    to the query. The overall cluster is further classified based on these partial score
    '''

    cluster: ChainCluster
    query: Query
    sim: float #Similarity to query
    candidates: list[SummaryCandidate] = field(default=None) #Looks confusing, but it's essentially the chains of the cluster, sorted by score
    model: CrossEncoder = field(default=None)

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

        self.candidates = [SummaryCandidate(chain, score, model) for score, chain in sorted(temp, reverse=True)]

    #---------------------------------------------------------------------------
    
    def evaluate_chains(self) -> 'SelectedCluster':
        '''
        Calculates the cross-encoder similarity score between the query and each chain in the cluster.
        After calling, scores are stored in ```self.chain_scores``` in the same order as the chains

        Arguments
        ---
        model: CrossEncoder
            The cross-encoder for the evaluation
        '''
        scores = self.model.predict([(self.query.text, c.text) for c in self.cluster.chains])
        self.candidates = [SummaryCandidate(chain, score) for score, chain in sorted(zip(scores, self.cluster.chains), reverse=True)]

        return self
    
    #---------------------------------------------------------------------------

    def chains(self) -> list[SentenceChain]:
        '''
        List of the releavance-sorted chains in descending order
        '''
        return list(chain.from_iterable([c.chains for c in self.candidates]))
    
    #---------------------------------------------------------------------------
    
    def scores(self) -> list[float]:
        '''
        List of the chain scores in descending order
        '''
        return [c.score for c in self.candidates]
    
    #---------------------------------------------------------------------------
    
    @property
    def cross_score(self) -> 'SelectedCluster':
        '''
        A relevance score for the entire cluster, by summing up the individual cross-encoder scores of the chains
        '''
        if self.candidates is None:
            return None

        return np.round(sum([score for score in self.scores()]), decimals=3)

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
        return self.cluster.text
    
    @property
    def clustering_context(self) -> ChainClustering:
        return self.cluster.clustering_context
    
    def __len__(self):
        return len(self.cluster)
        
    def __iter__(self):
        return iter(self.scored_chains)
    
#================================================================================================================

@dataclass
class SummarySegment():
    '''
    Represents a part/paragraph of the summary referring to one specific document.
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
            selected_clusters.append(SelectedCluster(sorted_clusters[i][1], query, sorted_clusters[i][0]))
    elif method == "thres":
        thres = 0.5
        for cluster in sorted_clusters:
            if cluster[0] > thres:
                cluster[2] = 11
                selected_clusters.append(SelectedCluster(cluster[1], query, cluster[0]))
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
    focused_cluster = selected_clusters[args.c]
    focused_cluster: SelectedCluster

    panel_print([f"Cluster [green]{cluster.id}[/green] with score [cyan]{cluster.sim:.3f}[/cyan]" for cluster in selected_clusters], title="Retrieved clusters based on cosine similarity")

    #Print cluster stats
    if args.stats:
        panel_print([Pretty(cluster_stats(cluster)) for cluster in selected_clusters], title="Cluster Stats")

    #Print cluster chains
    if args.print:
        focused_cluster.print()

    #Let's evaluate chains
    #===============================================================================================================
    cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2')

    #Calculate the cross-encoder socres
    #--------------------------------------------------------------------------
    os.makedirs("cross_scores", exist_ok=True)
    for cluster in selected_clusters:
        cluster.model = cross_model
        if not os.path.exists(f"cross_scores/{cluster.id}.pkl"):
            cluster.evaluate_chains()
            cluster.store_scores("cross_scores")
        else:
            cluster.load_scores("cross_scores")

    #Print cross-scores
    #--------------------------------------------------------------------------
    #Of all clusters
    panel_print([f"Cluster {cluster.id} score: [cyan]{cluster.cross_score:.3f}[/cyan]" for cluster in selected_clusters], title="Cross-encoder scores of the selected clusters")
    
    #For each chain separately
    panel_print(
            [f"Chain [green]{candidate.chain.index:03}[/green] with score [cyan]{candidate.score:.3f}[/cyan]" for candidate in focused_cluster.candidates] + 
            [Rule(), f"Cluster score: [cyan]{focused_cluster.cross_score:.3f}[/cyan]"],
        title=f"For cluster {focused_cluster.id}")

    print(focused_cluster.clustering_context.chains[44].text)