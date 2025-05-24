from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from mypackage.elastic import Document, ElasticDocument
from functools import cached_property
import json
from itertools import chain
from .helper import split_to_sentences
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..clustering.classes import ChainCluster

#============================================================================================

class SentenceLike(ABC):
    '''
    Represents an arbitrarily long text sequence. Can be a single sentence or multiple consecutive sentences.
    '''
    @property
    @abstractmethod
    def vector(self):
        pass

    @property
    @abstractmethod
    def text(self):
        pass

    def similarity(self, other) -> float:
        return np.dot(self.vector, other.vector)/(np.linalg.norm(self.vector)*np.linalg.norm(other.vector))

#============================================================================================    
    
@dataclass(repr=False)
class SimilarityPair:
    '''
    Pair of SentenceLike objects, along with their similarity score
    '''
    s1: SentenceLike
    s2: SentenceLike
    sim: float

    @classmethod
    def from_sentences(cls, s1: SentenceLike, s2: SentenceLike):
        return cls(s1, s2, s1.similarity(s2))

    def __post_init__(self):
        if not isinstance(self.s1, SentenceLike):
            raise ValueError("s1 must be a Sentence or SentenceChain")
        
        if not isinstance(self.s2, SentenceLike):
            raise ValueError("s2 must be a Sentence or SentenceChain")
    
#============================================================================================

@dataclass(repr=False)
class Sentence(SentenceLike):
    '''
    Represents a single sentence
    '''

    _text: str
    _vector: ndarray
    doc: Document
    offset: int = field(default=-1)
    parent_chain: "SentenceChain" = field(default=None)

    def __str__(self):
        return self.text
    
    def __len__(self) -> int:
        return len(self._text.split())
    
    @property
    def vector(self):
        return self._vector
    
    @property
    def text(self):
        return self._text
    
    #For wherever a numpy array is required as input
    def __array__(self, dtype=None):
        return self._vector
    
    #----------------------------------------------------------------------------------------------

    def next(self, n: int = 1, *, force_list: bool = False) -> Union['Sentence', list['Sentence']]:
        #print(f"{'-'*60}\nnext_sentence(n={n})")
        if n < 1:
            if force_list:
                return [self]
            return self
        #The parent chain is responsible for returning the most sentences it can
        #It will return the max number of sentences that it owns, and it will delegate responsibility
        #to the next chain for the remaining sentences
        return self.parent_chain.get_next_sentences(self.offset, n, force_list=force_list)
    
    #----------------------------------------------------------------------------------------------
    
    def prev(self, n: int = 1, *, force_list: bool = False) -> Union['Sentence', list['Sentence']]:
        print(f"{'-'*60}\nprev_sentence(n={n})")
        if n < 1:
            if force_list:
                return [self]
            return self
        #The parent chain is responsible for returning the most sentences it can
        #It will return the max number of sentences that it owns, and it will delegate responsibility
        #to the previous chain for the remaining sentences
        return self.parent_chain.get_prev_sentences(self.offset, n, force_list=force_list)
    
#============================================================================================
    
class SentenceChain(SentenceLike):
    '''
    Represents a chain of one or more consecutive sentences that are very similar
    '''
    _vector: ndarray
    sentences: list[Sentence]
    pooling_method: str
    parent_cluster: "ChainCluster"
    index: int #The chain index in the global list of chains

    EXEMPLAR_BASED_METHODS = []

    def __init__(self, sentences: SentenceLike | list[SentenceLike], pooling_method: str = "average", *, normalize: bool = True):
        '''
        Arguments
        ---
        normalize: bool
            Normalize the representative after pooling. Defaults to ```True```
        '''
        #REMEMBER: Never mix Sentence and SentenceChain in the same iterable

        self.index = None
        self.parent_cluster = None
        self.pooling_method = pooling_method

        #Convert single sentence into a list with only one sentence (or chain)
        #Don't worry, if it's a chain, we will extract its actual sentences below
        if isinstance(sentences, SentenceLike):
            sentences = [sentences]
        
        self._vector = SentenceChain.pooling(sentences, pooling_method, normalize=normalize)

        #Extract sentences from 'sentences' argument
        if isinstance(sentences[0], Sentence):
            self.sentences = sentences
        elif isinstance(sentences[0], SentenceChain):
            #Iterate over the sentences in the chains
            #Combine them all (in order) as part of the new chain
            #Obvisouly this assumes that you're not a dumbass and didn't mix Sentences and SentenceChains
            self.sentences = list(chain.from_iterable(sentences))

    @staticmethod
    def pooling_average(sentences: list[SentenceLike], *, normalize: bool = True) -> ndarray:
        vec = np.average(np.row_stack([s.vector for s in sentences]), axis=0)
        return vec / np.linalg.norm(vec) if normalize else vec
    
    @staticmethod
    def pooling_max(sentences: list[SentenceLike], *, normalize: bool = True) -> ndarray:
        vec = np.max(np.row_stack([s.vector for s in sentences]), axis=0)
        return vec / np.linalg.norm(vec) if normalize else vec

    @staticmethod
    def pooling(sentences: list[SentenceLike], pooling_method: str, *, normalize: bool = True) -> ndarray:
        '''
        Arguments
        ---
        normalize: bool
            Normalize the representative after pooling. Defaults to ```True```
        '''

        #But, there was nothing to pool
        if len(sentences) == 1:
            return sentences[0].vector

        match pooling_method:
            case "average": return SentenceChain.pooling_average(sentences, normalize=normalize)
            case "max": return SentenceChain.pooling_max(sentences, normalize=normalize)
    
    def sentence_matrix(self) -> ndarray:
        '''
        Converts the sentence chain (list) into a matrix, where each row is a sentence. Order is maintained
        '''
        return np.array([x.vector for x in self.sentences])

    def __iter__(self):
        return iter(self.sentences)
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    #For wherever a numpy array is required as input
    def __array__(self, dtype=None):
        return self._vector
    
    def __getitem__(self, key:int) -> Sentence:
        return self.sentences[key]
    
    def get_global(self, global_key: int) -> Sentence:
        if global_key not in self.offset_range:
            raise ValueError(f"The global sentence index {global_key} is not in the chain's range {self.offset_range}")

        return self.sentences[global_key - self.offset]
    
    #--------------------------------------------------------------------------------------------------------------------------
    
    def get_next_sentences(self, offset: int, n: int = 1, *, force_list: bool = False) -> Sentence | list[Sentence]:
        '''
        Arguments
        ---
        offset: int
            The offset of the sentence that is requesting the next sentences.
            In other words, this is NOT the offset of the first retrieved sentence, but the one before
        n: int
            The number of next sentences to retrieve. Defaults to ```1```

        force_list: bool
            Force the result to be returned as a list, even for a single sentence. Defauls to ```False```

        Returns
        ---
        sentences: Sentence | list[Sentence]
            A list with the next ```n``` sentences. If ```n==1```, then only one ```Sentence``` is returned,
            unless ```force_list==True```
        '''
        #print(f"get_next_sentences(requester_offset={offset}, n={n}) len={self.__len__()}")

        if offset + 1 < self.offset:
            raise ValueError(f"Chain with offset {self.offset} cannot provide sentences starting from offset {offset}")

        last_owned_offset = self.offset + self.__len__() - 1
        max_owned_offset_from_requested = min(offset + n, last_owned_offset)
        remaining_sentences = max(0, offset + n - last_owned_offset)

        #print(f"last_owned_offset = {last_owned_offset}")
        #print(f"max_owned_offset_from_requested = {max_owned_offset_from_requested}")
        #print(f"remaining_sentences = {remaining_sentences}")

        #Sentences from the next chain
        extra_sentences = []

        if remaining_sentences > 0:
            #Το 'πα εγώ στον σκύλο μου κι' ο σκύλος στην ουρά του
            extra_sentences = self.next().get_next_sentences(last_owned_offset, remaining_sentences, force_list=True)

        #print(f"start={offset + 1 - self.offset}")
        #print(f"end={max_owned_offset_from_requested - self.offset + 1}")

        result = self.sentences[(offset + 1 - self.offset) : (max_owned_offset_from_requested - self.offset + 1)] + extra_sentences

        if force_list:
            return result
        
        return result if n > 1 else result[0]
    
    #--------------------------------------------------------------------------------------------------------------------------
    
    def get_prev_sentences(self, offset: int, n: int = 1, *, force_list: bool = False) -> Sentence | list[Sentence]:
        '''
        Arguments
        ---
        offset: int
            The offset of the sentence that is requesting the previous sentences.
            In other words, this is NOT the offset of the last retrieved sentence, but the one after
        n: int
            The number of next sentences to retrieve. Defaults to ```1```

        force_list: bool
            Force the result to be returned as a list, even for a single sentence. Defauls to ```False```

        Returns
        ---
        sentences: Sentence | list[Sentence]
            A list with the previous ```n``` sentences. If ```n==1```, then only one ```Sentence``` is returned,
            unless ```force_list==True```
        '''
        print(f"get_prev_sentences(requester_offset={offset}, n={n}) len={self.__len__()}")

        if offset + 1 < self.offset:
            raise ValueError(f"Chain with offset {self.offset} cannot provide sentences starting from offset {offset}")

        first_owned_offset = self.offset
        min_owned_offset_from_requested = max(offset - n, first_owned_offset)
        remaining_sentences = max(0, first_owned_offset - offset + n)

        print(f"first_owned_offset = {first_owned_offset}")
        print(f"min_owned_offset_from_requested = {min_owned_offset_from_requested}")
        print(f"remaining_sentences = {remaining_sentences}")

        #Sentences from the next chain
        extra_sentences = []

        if remaining_sentences > 0:
            #Το 'πα εγώ στον σκύλο μου κι' ο σκύλος στην ουρά του
            extra_sentences = self.prev().get_prev_sentences(first_owned_offset, remaining_sentences, force_list=True)

        print(f"start={min_owned_offset_from_requested - self.offset}")
        print(f"end={offset - self.offset}")

        result = extra_sentences + self.sentences[(min_owned_offset_from_requested - self.offset) : (offset - self.offset)]

        if force_list:
            return result
        
        return result if n > 1 else result[0]
    
    #--------------------------------------------------------------------------------------------------------------------------

    def next(self) -> 'SentenceChain':
        '''
        Returns the next chain of the document, regardless of the cluster it belongs to.
        A clustering context must exist for this to work.
        '''
        #print(f"next_chain(self.index={self.index})")
        return self.parent_cluster.clustering_context.chains[self.index + 1]
    
    #--------------------------------------------------------------------------------------------------------------------------
    
    def prev(self) -> 'SentenceChain':
        '''
        Returns the previous chain of the document, regardless of the cluster it belongs to.
        A clustering context must exist for this to work.
        '''
        #print(f"previous_chain(self.index={self.index})")
        return self.parent_cluster.clustering_context.chains[self.index - 1]
    
    #--------------------------------------------------------------------------------------------------------------------------

    def __str__(self):
        return f"SentenceChain(start_offset={self.offset}, size={self.__len__()}, end_offset={self.offset + self.__len__() - 1})"
    
    @property
    def vector(self):
        '''
        Get the representative vector of this chain
        '''
        return self._vector
    
    @cached_property
    def text(self):
        #For hierarchical chain clustering, this is essentially recursive
        #Similar to an in-order traversal
        return " ".join([s.text for s in self.sentences])
    
    @property
    def doc(self) -> Document:
        return self.sentences[0].doc
    
    @property
    def offset(self) -> int:
        return self.sentences[0].offset
    
    @property
    def offset_range(self) -> range:
        return range(self.offset, self.offset + self.__len__())

    def data(self) -> dict:
        return {
            'vector': self.vector,
            'offset': self.offset,
            'pooling_method': self.pooling_method,
            'index': self.index,
            'sentences': [s.vector for s in self.sentences]
        }
    
    @classmethod
    def from_data(cls, data: dict, doc: Document, *, parent: "ChainCluster" = None) -> 'SentenceChain':
        obj = cls.__new__(cls)
        obj._vector = data['vector']
        obj.pooling_method = data['pooling_method']
        obj.parent_cluster = parent
        obj.index = data.get('index', None)

        offset = data['offset']
        text = split_to_sentences(doc.text)
        obj.sentences = [Sentence(text[offset + i], vec, doc, offset + i, parent_chain=obj) for i, vec in enumerate(data['sentences'])]

        return obj