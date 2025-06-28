import os
import json
import numpy as np
from itertools import chain
from dataclasses import dataclass, field
from transformers import LlamaTokenizer
import time

from ..cluster_selection import SelectedCluster, SummaryCandidate
from ..elastic import ElasticDocument, Document
from ..llm import LLMSession, llm_summarize
from ..helper import panel_print
from ..query import Query
from ..sentence import SentenceChain

#================================================================================================================

@dataclass
class SummarySegment():
    '''
    The input and output of the summarization system.
    Represents the part/paragraph of the final summary referring to one specific document.
    Contains all the retrieved clusters that are relevant from the specific document, as well as general information
    about the document itself that should be included in the summary (e.g. author and main topic)


    NOTE: Maybe this isn't that good. Maybe I should not be grouping based on document,
    since I no longer summarize each document separately
    '''
    clusters: list[SelectedCluster]
    doc: ElasticDocument
    #Could either be the paper's existing summary, or an LLM-generated text of the first paragraph or summary
    #...maybe it could even be pre-calculated?
    extra_info: str = field(default=None) 
    summary: Document = field(default=None)
    #One list element for each sentence in the summary.
    #Points to the citation, which is a position in flat_chains list.
    #None means no citation for this sentence
    citations: list[int] = field(default=None)
    created_with: str = field(default=None)

    #---------------------------------------------------------------------------------------------------

    @property
    #Unsure if this is useful
    def cross_score(self) -> float:
        '''
        Just like the SelectedClusters, the summary segments also have a cross-score,
        which is the sum of the clusters scores. This is only an approximation, as different clusters
        of the same document can potentially have overlapping candidates (after context expansion),
        which affect the score
        '''
        return np.sum(c.cross_score for c in self.clusters)

    #---------------------------------------------------------------------------------------------------

    def flat_candidates(self) -> list[SummaryCandidate]:
        '''
        Returns a list of all the best candidates from each selected clusters of the document, in order of appearance
        '''
        flat_candidates: list[SummaryCandidate] = []
        for c in self.clusters:
            flat_candidates += c.selected_candidates()
        #Sort by start
        return sorted(flat_candidates, key=lambda x: x.index_range.start, reverse=False)
    
    #---------------------------------------------------------------------------------------------------

    def flat_chains(self) -> list[SentenceChain]:
        '''
        Turns the chains from the ordered candidates into one single matrix
        We assume that merge_candidates has been called on the selected clusters, so that overlaps are resolved
        '''
        flat_candidates: list[SummaryCandidate] = []
        for c in self.clusters:
            flat_candidates += c.selected_candidates()
        #Sort by start
        temp = sorted(flat_candidates, key=lambda x: x.index_range.start, reverse=False)

        #Handle duplicate chains (because same chain appears in positive and negative contexts)
        flat_chains = []
        prev = -1
        for x in chain.from_iterable([c.context.chains for c in temp]):
            if prev != x.index:
                flat_chains.append(x)
                prev = x.index
            else:
                print(x.index)

        return flat_chains
    
    #---------------------------------------------------------------------------------------------------

    @property
    def text(self):
        '''
        Get the text to be summarized, by combining all the chains from the selected clusters in order of apperance.
        Reminder: the chains that are returned from each cluster depend on how relevant they are to the query
        '''
        flat_candidates: list[SummaryCandidate] = []
        for c in self.clusters:
            flat_candidates += c.selected_candidates()
        
        #Sort by start
        temp = sorted(flat_candidates, key=lambda x: x.index_range.start, reverse=False)

        #NOTE: What happens if two candidates from different clusters have overlapping chains??

        return "\n\n".join([c.text for c in temp])
    
    #---------------------------------------------------------------------------------------------------

    def pretty_text(self, *, show_added_context = False, show_chain_indices = False, show_chain_sizes = False):
        '''
        Get the text to be summarized, by combining all the chains from the selected clusters in order of apperance.
        Reminder: the chains that are returned from each cluster depend on how relevant they are to the query
        '''
        flat_candidates: list[SummaryCandidate] = []
        for c in self.clusters:
            flat_candidates += c.selected_candidates()
        
        #Sort by start
        temp = sorted(flat_candidates, key=lambda x: x.index_range.start, reverse=False)

        #NOTE: What happens if two candidates from different clusters have overlapping chains??

        return "\n\n".join([c.pretty_text(show_added_context=show_added_context, show_chain_indices=show_chain_indices, show_chain_sizes=show_chain_sizes) for c in temp])
    
    #---------------------------------------------------------------------------------------------------
    
    @property
    def id(self) -> str:
        return f"{self.doc.id:04}_" + "_".join([f"{cluster.cluster.label:02}" for cluster in self.clusters])
    
    #---------------------------------------------------------------------------------------------------
    
    def summary_matrix(self) -> np.ndarray:
        '''
        Converts the summary sentences into a matrix where each row is a sentence embedding. Order is maintained
        '''
        return np.array([x.vector for x in self.summary.sentences])
        
    #---------------------------------------------------------------------------------------------------

    def store_summary(self):
        os.makedirs("generated_summaries", exist_ok=True)

        with open(f"generated_summaries/segment_{self.id}.json", "w") as f:
            json.dump({
                'created_with': self.created_with,
                'summary': self.summary.text
            }, f)

    #---------------------------------------------------------------------------------------------------

    def load_summary(self):
        if not os.path.exists(f"generated_summaries/segment_{self.id}.json"):
            return False
        
        with open(f"generated_summaries/segment_{self.id}.json", "r") as f:
            data = json.load(f)
            self.summary = Document(data['summary'])
            self.created_with = data['created_with']

        return True

    #---------------------------------------------------------------------------------------------------
    
    def __array__(self):
        return np.array(self.flat_chains())
    
#================================================================================================================

class Summarizer():
    '''
    Contains a connection to the LLM used for summarization.
    The final output is the summary
    '''
    llm: LLMSession
    query: Query

    def __init__(self, query: Query, *, llm: LLMSession = None):
        self.query = query
        if llm is None:
            self.llm = LLMSession()
        else:
            self.llm = llm
    
    #---------------------------------------------------------------------------------------------------

    def summarize_segments(self, segments: list[SummarySegment]):
        '''
        Yields
        ---
        fragment: str
            The next text fragment of the output summary
        '''
        text_to_summarize = "\n".join([f"Document {segment.doc.id}\n-----\n{segment.text}" for segment in segments if len(segment.text.split()) > 0])
        #tokens = LlamaTokenizer.from_pretrained("huggyllama/llama-7b").tokenize(text_to_summarize)
        #print(len(tokens))
        yield from llm_summarize(self.llm, self.query.text, text_to_summarize)


