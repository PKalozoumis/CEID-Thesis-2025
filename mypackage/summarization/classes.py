from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cluster_selection import SelectedCluster, SummaryCandidate

import os
import json
import numpy as np
from itertools import chain
from dataclasses import dataclass, field
from transformers import LlamaTokenizer
import time
from collections import defaultdict
import re
import copy

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

    sorted_candidates: list[SummaryCandidate]
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
            clusters_per_doc.sort(key=lambda cluster_list: sum(c.cross_score for c in cluster_list), reverse=True)

            self.sorted_candidates = []

            #From each cluster, extract selected candidates and sort them by either relevance or appearance
            for cluster_list in clusters_per_doc:
                local_candidates = list(chain.from_iterable(c.candidates for c in cluster_list))
                if 'relevance' in self.sorting_method:
                    self.sorted_candidates += sorted(local_candidates, key=lambda c: c.score, reverse=True)
                else:
                    self.sorted_candidates += sorted(local_candidates, key=lambda c: c.first_index, reverse=False)
        #------------------------------------------------------------------------------------

        elif self.sorting_method.startswith("cluster"):
            #Sort clusters by their selected candidate cross-score
            sorted_clusters = sorted(clusters, key=lambda c: c.cross_score, reverse=True)

            self.sorted_candidates = []

            #From each cluster, extract selected candidates and sort them by either relevance or appearance
            for cluster in sorted_clusters:
                if 'relevance' in self.sorting_method:
                    self.sorted_candidates += sorted(cluster.candidates, key=lambda c: c.score, reverse=True)
                else:
                    self.sorted_candidates += sorted(cluster.candidates, key=lambda c: c.first_index)

        #------------------------------------------------------------------------------------
        elif self.sorting_method == "flat_relevance":
            self.sorted_candidates = sorted(chain.from_iterable([cluster.candidates for cluster in clusters]), key=lambda candidate: candidate.score, reverse=True)

        self.sorted_candidates = self._resolve_overlaps(self.sorted_candidates)
    
    #------------------------------------------------------------------------------------

    @staticmethod
    def _resolve_overlaps(candidates: list[SummaryCandidate]) -> list[SummaryCandidate]:
            #Give each element a global position in the sorted list of candidates
            #This will help us resolve overlaps in a single document, while also maintaining the original sort order
            candidates: list[tuple[int, SummaryCandidate]] = list(enumerate(candidates))

            #Group candidates by document
            groups = defaultdict(list)
            for pos,c in candidates:
                groups[c.chain.doc.id].append((pos,c))
            groups: list[list[tuple[int, SummaryCandidate]]] = list(groups.values())

            marked_for_deletion = [False] * len(candidates)

            #Resolve overlaps for each group
            for group in groups:
                seen_chains = set()
                for pos, c in group:
                    index_set = set(c.index_range)
                    if seen_chains & index_set:
                        marked_for_deletion[pos] = True
                    else:
                        seen_chains |= index_set
                
            return [c for pos,c in candidates if not marked_for_deletion[pos]]
    
    #------------------------------------------------------------------------------------

    @property
    def query(self) -> Query:
        return self.clusters[0].evaluator.query
    
    #------------------------------------------------------------------------------------
    
    def clean_summary(self, no_citations: bool = False, no_newlines: bool = False, inplace: bool = False) -> SummaryUnit:
        new_obj = copy.copy(self)

        if self.summary is not None:
            if no_citations:
                new_obj.summary = re.sub(r"<\d+_\d+-\d+>", "", new_obj.summary)
            if no_newlines:
                new_obj.summary = re.sub(r"\n+", " ", new_obj.summary)

        if inplace:
            self.summary = new_obj.summary
            return self
        else:
            return new_obj
    
    #------------------------------------------------------------------------------------

    def pretty_print(self, *, show_added_context = False, show_chain_indices = False, show_chain_sizes = False, return_text = True, console_width=None):

        if len(self.sorted_candidates) == 0:
            panel = panel_print("But, there was nothing to print", title="Summarization input text (formatted)", return_panel=return_text)
        else:
            to_print = []

            for candidate in self.sorted_candidates:
                id = candidate.citation
                #id = f"<{candidate.chain.doc.id}_{candidate.first_sentence_index}-{candidate.last_sentence_index}>"
                #FF6A00 -> orange
                #FF64DC -> pink
                to_print += [f"[#FF6A00]{id}[/#FF6A00] [#FF64DC]({candidate.score:.3f})[/#FF64DC] [red]->[/red] {candidate.pretty_text(show_added_context=show_added_context, show_chain_indices=show_chain_indices, show_chain_sizes=show_chain_sizes)}\n"]

            panel = panel_print(to_print, title="Summarization input text (formatted)", return_panel=return_text)

        #----------------------------------
        
        if return_text:
            return rich_console_text(panel, console_width=console_width)

    #------------------------------------------------------------------------------------

    @property
    def text(self) -> str:
        txt = ""
        for candidate in self.sorted_candidates:
            txt += f"<{candidate.chain.doc.id}_{candidate.first_sentence_index}-{candidate.last_sentence_index}>: {candidate.text}\n\n"
        return txt
    
    #------------------------------------------------------------------------------------

    @property
    def reference(self) -> str:
        txt = ""
        for candidate in self.sorted_candidates:
            txt += candidate.text
        return txt
    
    #------------------------------------------------------------------------------------

    def data(self) -> dict:
        return {
            'summary': self.summary,
            'citations': self.citations,
            'sorting_method': self.sorting_method
            #Sorted candidates are not included, because they can be recreated from the selected clusters and the sorting method
        }
    
    @classmethod
    def from_data(cls, data, clusters: list[SelectedCluster]) -> 'SummaryUnit':
        temp = cls(clusters, data['sorting_method'])
        temp.summary = data['summary']
        temp.citations = data['citations']

        return temp

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

        prefix_regex = re.compile(r"(.*?)(<(\d+|$)(_(\d+|$))?(-(\d+|$))?(>|$))(.*)")
        full_regex = re.compile(r"(.*?)(<(?P<doc>\d+)_(?P<start>\d+)-(?P<end>\d+)>)(.*)")

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



            


