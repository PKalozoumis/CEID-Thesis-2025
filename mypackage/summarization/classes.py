from ..cluster_selection import SelectedCluster, SummaryCandidate
from ..elastic import ElasticDocument, Document
from ..llm import LLMSession, llm_summarize
from ..helper import panel_print
from ..query import Query
from ..sentence import SentenceChain
import os
import json
import numpy as np
from itertools import chain

from dataclasses import dataclass, field
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer, PegasusForConditionalGeneration, PreTrainedTokenizer

#================================================================================================================

@dataclass
class SummarySegment():
    '''
    The input and output of the summarization system.
    Represents the part/paragraph of the final summary referring to one specific document.
    Contains all the retrieved clusters that are relevant from the specific document, as well as general information
    about the document itself that should be included in the summary (e.g. author and main topic)
    '''
    clusters: list[SelectedCluster]
    doc: ElasticDocument
    #Could either be the paper's existing summary, or an LLM-generated text of the first paragraph or summary
    #...maybe it could even be pre-calculated?
    extra_info: str = field(default=None) 
    #summary: Document = field(default=Document.with_schema({'summary': str, 'segment_obj': object}, "summary"))
    summary: Document = field(default=Document.with_schema(str))
    #One list element for each sentence in the summary.
    #Points to the citation, which is a position in flat_chains list.
    #None means no citation for this sentence
    citations: list[int] = field(default=None)
    created_with: str = field(default=None)

    #---------------------------------------------------------------------------------------------------

    @property
    def text(self):
        flat_candidates: list[SummaryCandidate] = []
        for c in self.clusters:
            flat_candidates += c.selected_candidates()
        
        #Sort by start
        temp = sorted(flat_candidates, key=lambda x: x.index_range.start, reverse=False)
        return "\n\n".join([c.text for c in temp])
    
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
                'summary': self.summary
            }, f)

    #---------------------------------------------------------------------------------------------------

    def load_summary(self):
        if not os.path.exists(f"generated_summaries/segment_{self.id}.json"):
            return False
        
        with open(f"generated_summaries/segment_{self.id}.json", "r") as f:
            data = json.load(f)
            #self.summary.doc = {'summary': data['summary'], 'segment_obj': self}
            self.summary.doc = data['summary']
            self.created_with = data['created_with']

        return True
    
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

        return flat_chains

    #---------------------------------------------------------------------------------------------------
    
    def __array__(self):
        return np.array(self.flat_chains())
    
#================================================================================================================

class Summarizer():
    '''
    Contains all the models necessary for summarization. Also contains a connection to the LLM.
    Classifies the provided ```SummarySegments``` and summarizes them using the appropriate model.
    The final output is the summary
    '''
    sus: PegasusForConditionalGeneration #𐐘
    megasus: BigBirdPegasusForConditionalGeneration #ඞ
    sus_tokenizer: PreTrainedTokenizer
    megasus_tokenizer: PreTrainedTokenizer
    llm: LLMSession
    query: Query

    def __init__(self, query: Query, *, sus: str = "google/pegasus-pubmed", megasus: str = "google/bigbird-pegasus-large-pubmed", llm: LLMSession = None):
        self.query = query
        self.sus = PegasusForConditionalGeneration.from_pretrained(sus)
        self.megasus = BigBirdPegasusForConditionalGeneration.from_pretrained(megasus)
        #self.sus = None
        #self.megasus = None
        self.sus_tokenizer = AutoTokenizer.from_pretrained(sus)
        self.megasus_tokenizer = AutoTokenizer.from_pretrained(megasus)
        if llm is None:
            self.llm = LLMSession()
        else:
            self.llm = llm
        
    #---------------------------------------------------------------------------------------------------

    def summarize_single_segment(self, segment: SummarySegment) -> SummarySegment:
        if len(segment.text.split()) < 100:
            segment.summary = segment.text
            segment.created_with = "same"
        elif len(segment.text.split()) < 900:
            #Use LLM
            '''
            inputs = self.sus_tokenizer(segment.text, return_tensors='pt', truncation=True, max_length=1024)
            prediction = self.sus.generate(**inputs)
            prediction = self.sus_tokenizer.batch_decode(prediction, skip_special_tokens=True)
            panel_print(prediction)
            '''
            txt = ""
            for fragment in llm_summarize(self.llm, self.query.text, segment.text):
                txt += fragment

            segment.summary = txt
            segment.created_with = "llm"
        else:
            #Use BigBird
            inputs = self.megasus_tokenizer(segment.text, return_tensors='pt', truncation=True, max_length=4096)
            prediction = self.megasus.generate(**inputs)
            prediction = self.megasus_tokenizer.batch_decode(prediction, skip_special_tokens=True)
            segment.summary = prediction[0]
            segment.created_with = "bigbird"
        return segment
    
    #---------------------------------------------------------------------------------------------------

    def summarize_segments(self, segments: list[SummarySegment]) -> list[SummarySegment]:
        for segment in segments:
            if not segment.load_summary():
                print("Generating summary...")
                self.summarize_single_segment(segment).store_summary()
        return segments

