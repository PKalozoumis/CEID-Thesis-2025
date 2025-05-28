from ..cluster_selection import SelectedCluster
from ..elastic import ElasticDocument

from dataclasses import dataclass, field
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer, PegasusForConditionalGeneration

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