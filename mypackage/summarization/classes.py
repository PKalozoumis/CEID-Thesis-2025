import os
import json
import numpy as np
from itertools import chain
from dataclasses import dataclass, field
from transformers import LlamaTokenizer
import time
from collections import defaultdict
import re

from ..cluster_selection import SelectedCluster, SummaryCandidate
from ..elastic import ElasticDocument, Document
from ..llm import LLMSession
from ..helper import panel_print, rich_console_text
from ..query import Query
from ..sentence import SentenceChain

from rich.rule import Rule
from rich.padding import Padding

#================================================================================================================

class SummaryUnit():
    '''
    The input and ouput of the summarization system. Contains all the candidates, as well as the final summary with its citations
    '''

    sorted_candidates: list[list[SummaryCandidate]] #What this represents depends on the sorting method
    summary: str
    citations: list[int]
    clusters: list[SelectedCluster]
    sorting_method: str
    

    def __init__(self, clusters: list[SelectedCluster], sorting_method: str):
        '''
        The input and ouput of the summarization system. Contains all the candidates, as well as the final summary with its citations

        Arguments
        ---
        clusters: list[SelectedCluster]
            The clusters that contain the candidates to be summarized.
        sorting_method: str
            Describes the order in which the candidates will be included in the summarization input. Possible values are:
            - ```flat_relevance```: All candidates are sorted in decreasing order of relevance, regardless of document
            - ```document_relevance```: Documents are sorted based on the selected candidate cross-score of their clusters.
            Within each document, candidates appear in order of relevance
            - ```document_appearance```: Documents are sorted based on the selected candidate cross-score of their clusters.
            Within each document, candidates appear in order of appearance
            - ```cluster_relevance```: Clusters are sorted based on their selected candidate cross-score.
            Within each cluster, candidates appear in order of relevance
            - ```cluster_appearance```: Clusters are sorted based on their selected candidate cross-score.
            Within each cluster, candidates appear in order of appearance
        '''
        #Best options are probably flat_relevance or document_relevance

        self.clusters = clusters
        self.summary = None
        self.citations = None
        self.sorting_method = sorting_method

        #------------------------------------------------------------------------------------

        if self.sorting_method.startswith("document"):
            #Group clusters by document
            clusters_per_doc = defaultdict(list)
            for focused_cluster in clusters:
                clusters_per_doc[focused_cluster.cluster.doc].append(focused_cluster)

            clusters_per_doc = [cluster_list for cluster_list in clusters_per_doc.values()]
            clusters_per_doc: list[list[SelectedCluster]]

            #Sort document lists by the sum of cluster scores
            clusters_per_doc.sort(key=lambda cluster_list: sum(c.selected_candidate_cross_score for c in cluster_list), reverse=True)

            #From each cluster, extract selected candidates and sort them by either relevance or appearance
            self.sorted_candidates = [
                sorted(
                    chain.from_iterable([c.selected_candidates() for c in cluster_list]),
                    key=lambda candidate: candidate.score,
                    reverse=True
                ) if 'relevance' in self.sorting_method else sorted (
                    chain.from_iterable([c.selected_candidates() for c in cluster_list]),
                    key=lambda candidate: candidate.first_index,
                    reverse=False
                )
                for cluster_list in clusters_per_doc
            ]

        #------------------------------------------------------------------------------------

        elif self.sorting_method.startswith("cluster"):
            #Sort clusters by their selected candidate cross-score
            sorted_clusters = sorted(clusters, key=lambda c: c.selected_candidate_cross_score, reverse=True)

            #From each cluster, extract selected candidates and sort them by either relevance or appearance
            self.sorted_candidates = [
                sorted(
                    cluster.selected_candidates(),
                    key=lambda candidate: candidate.score,
                    reverse=True
                ) if 'relevance' in self.sorting_method else sorted (
                    cluster.selected_candidates(),
                    key=lambda candidate: candidate.first_index,
                    reverse=False
                )
                for cluster in sorted_clusters
            ]

        #------------------------------------------------------------------------------------
        elif self.sorting_method == "flat_relevance":
            self.sorted_candidates = [sorted(chain.from_iterable([cluster.selected_candidates() for cluster in clusters]), key=lambda candidate: candidate.score, reverse=True)]


        self.sorted_candidates = list(filter(lambda x: len(x) > 0, self.sorted_candidates))
    
    #------------------------------------------------------------------------------------

    def pretty_print(self, *, show_added_context = False, show_chain_indices = False, show_chain_sizes = False, return_text = True, console_width=None):

        if len(self.sorted_candidates) == 0:
            panel = panel_print("But, there was nothing to print", title="Summarization input text (formatted)", return_panel=return_text)
        else:
            def candidate_list_pretty_text(candidate_list: list[SummaryCandidate]):
                return "\n\n".join([c.pretty_text(show_added_context=show_added_context, show_chain_indices=show_chain_indices, show_chain_sizes=show_chain_sizes) for c in candidate_list])

            to_print = []

            if self.sorting_method == "flat_relevance":
                for candidate in self.sorted_candidates[0]:
                    id = f"<{candidate.chain.doc.id}_{candidate.first_sentence_index}-{candidate.last_sentence_index}>"
                    to_print += [f"[#FF6A00]{id}[/#FF6A00] [#FF64DC]({candidate.score:.3f})[/#FF64DC] [red]->[/red] {candidate.pretty_text(show_added_context=show_added_context, show_chain_indices=show_chain_indices, show_chain_sizes=show_chain_sizes)}\n"]
            else:
                for candidate_list in self.sorted_candidates:
                    to_print.append(Rule(f"Document {candidate_list[0].chain.doc.id}" if self.sorting_method.startswith("document") else f"Cluster {candidate_list[0].chain.parent_cluster.id}"))
                    to_print += [candidate_list_pretty_text(candidate_list)]
                    to_print.append("")

            panel = panel_print(to_print, title="Summarization input text (formatted)", return_panel=return_text)

        #----------------------------------
        
        if return_text:
            return rich_console_text(panel, console_width=console_width)

    #------------------------------------------------------------------------------------

    @property
    def text(self) -> str:
        txt = ""
        for candidate_list in self.sorted_candidates:
            for candidate in self.sorted_candidates[0]:
                txt += f"<{candidate.chain.doc.id}_{candidate.first_sentence_index}-{candidate.last_sentence_index}>: {candidate.text}\n\n"
        return txt
    
#================================================================================================================

class Summarizer():
    '''
    Contains a connection to the LLM used for summarization.
    The final output is the summary
    '''
    llm: LLMSession
    query: Query

    def __init__(self, query: Query, llm=LLMSession):
        self.query = query
        self.llm = llm
    
    #---------------------------------------------------------------------------------------------------

    def summarize(self, unit: SummaryUnit, stop_dict, *, cache_prompt: bool = False):
        '''
        Yields
        ---
        stream: PredictionStream
            The stream from the LLM
        fragment: str
            The next text fragment of the output summary
        citation: dict|None
            If a citation is included at the end of the current fragment, an object is returned that described the citation.
            If there's no citation, None is returned
        '''

        unit.summary = ""
        parsing_citation = False
        citation_temp_text = ""

        prefix_regex = re.compile("(.*?)(<(\d+|$)(_(\d+|$))?(-(\d+|$))?(>|$))(.*)")
        full_regex = re.compile("(.*?)(<(?P<doc>\d+)_(?P<start>\d+)-(?P<end>\d+)>)(.*)")

        for fragment in self.llm.summarize(self.query.text, unit.text, stop_dict, cache_prompt=cache_prompt):

            #Identify start of citation
            if not parsing_citation:
                if res := prefix_regex.match(fragment):
                    parsing_citation = True
                    citation_temp_text = res.group(2) #We hold the text that is potentially a citation

                    unit.summary += res.group(1)
                    yield res.group(1), None #The first part (before the <) we can safely return to the user

                else: #Normal operations
                    unit.summary += fragment
                    yield fragment, None
            else:
                #We need to check if the new input still matches
                #If yes, then we should check if we're done
                citation_temp_text += fragment
                if prefix_regex.match(citation_temp_text):
                    #We are done. We found the entire citation
                    if res := full_regex.match(citation_temp_text):
                        
                        temp = temp = res.group(1) + "<citation>"

                        #What if the remaining text is also a citation?
                        res2 = prefix_regex.match(res.group(6))

                        #We are done
                        if res2 is None:
                            temp += res.group(6)
                            citation_temp_text = ""
                            parsing_citation = False
                        #Remaining text IS a potential citation. It should be examined next iteration
                        else:
                            citation_temp_text = res.group(6)

                        unit.summary += temp
                        yield temp, {
                            'doc': int(res.group('doc')),
                            'start': int(res.group('start')),
                            'end': int(res.group('end')) if res.group('end') is not None else int(res.group('start'))
                        }
                else: #The input stops being a citation. We throw the held text to the summary
                    unit.summary += citation_temp_text
                    yield citation_temp_text, None
                    citation_temp_text = ""



            


