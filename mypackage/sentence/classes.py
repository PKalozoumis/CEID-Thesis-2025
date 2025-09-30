from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, Union
from sklearn.metrics.pairwise import cosine_similarity
import warnings

from .helper import split_to_sentences
if TYPE_CHECKING:
    from ..elastic import Document
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
    def text(self) -> str:
        pass

    def similarity(self, other) -> float:
        return np.dot(self.vector, other.vector)/(np.linalg.norm(self.vector)*np.linalg.norm(other.vector))

#============================================================================================    
    
@dataclass(repr=False)
class SimilarityPair:
    '''
    Pair of ```SentenceLike``` objects, along with their similarity score

    Arguments
    ---
    s1: SentenceLike
        First object
    s2: SentenceLike
        Second object
    sim: float
        Their similarity
    '''
    s1: SentenceLike
    s2: SentenceLike
    sim: float

    #----------------------------------------------------------------------------------------------

    @classmethod
    def from_sentence_like(cls, s1: SentenceLike, s2: SentenceLike):
        return cls(s1, s2, s1.similarity(s2))
    
    #----------------------------------------------------------------------------------------------

    def __post_init__(self):
        if not isinstance(self.s1, SentenceLike):
            raise ValueError("s1 must be a Sentence or SentenceChain")
        
        if not isinstance(self.s2, SentenceLike):
            raise ValueError("s2 must be a Sentence or SentenceChain")
        
        #print(self.sim)
    
#============================================================================================

@dataclass(repr=False)
class Sentence(SentenceLike):
    '''
    Represents a single sentence

    Arguments
    ---
    _text: str
        The text contents of the sentence
    _vector: np.ndarray
        The sentence embedding
    doc: Document
        Document where the sentence belongs
    index: int, optional
        Sentence offset inside the document
    parent_chain: SentenceChain, optional
        The chain where the sentence belongs. The chaining algorithm must have been run. Otherwise, this is ```None```
    '''

    _text: str
    _vector: ndarray
    doc: Document
    index: int = field(default=-1) #The sentence index inside the document
    parent_chain: SentenceChain = field(default=None)

    #----------------------------------------------------------------------------------------------
    
    @property
    def vector(self):
        return self._vector
    
    @property
    def text(self):
        return self._text
    
    #----------------------------------------------------------------------------------------------

    def next(self, n: int = 1, *, force_list: bool = False) -> Union[Sentence, list[Sentence]]:
        '''
        Returns the next ```n``` sentences from the document where this sentence belongs

        Arguments
        ---
        n: int
            The number of sentences to return. Defaults to ```1```
        force_list: bool
            If set to ```True```, the resulting sentences are always returned as a list, even when 
            we only requested one sentence. Defaults to ```False```
        '''
        if n < 1:
            if force_list:
                return [self]
            return self
        #The parent chain is responsible for returning the most sentences it can
        #It will return the max number of sentences that it owns, and it will delegate responsibility
        #to the next chain for the remaining sentences
        return self.parent_chain.get_next_sentences(self.index, n, force_list=force_list)
    
    #----------------------------------------------------------------------------------------------
    
    def prev(self, n: int = 1, *, force_list: bool = False) -> Union[Sentence, list[Sentence]]:
        '''
        Returns the previous ```n``` sentences from the document where this sentence belongs

        Arguments
        ---
        n: int
            The number of sentences to return. Defaults to ```1```
        force_list: bool
            If set to ```True```, the resulting sentences are always returned as a list, even when 
            we only requested one sentence. Defaults to ```False```

        Returns
        ---
        '''
        if n < 1:
            if force_list:
                return [self]
            return self
        #The parent chain is responsible for returning the most sentences it can
        #It will return the max number of sentences that it owns, and it will delegate responsibility
        #to the previous chain for the remaining sentences
        return self.parent_chain.get_prev_sentences(self.index, n, force_list=force_list)
    
    #----------------------------------------------------------------------------------------------
    
    #For wherever a numpy array is required as input
    def __array__(self, dtype=None):
        return self._vector

    def __str__(self):
        return self.text
    
    def __len__(self) -> int:
        return len(self._text.split())
    
#============================================================================================
    
class SentenceChain(SentenceLike):
    '''
    Represents a chain of one or more consecutive sentences that are sufficiently similar
    '''
    _vector: ndarray
    sentences: list[Sentence]
    pooling_method: str
    parent_cluster: ChainCluster
    index: int #Chain order inside the document

    VALID_METHODS = ["average", "max", "most_similar", "k_most_similar"]
    EXEMPLAR_BASED_METHODS = ["most_similar", "k_most_similar"]

    #----------------------------------------------------------------------------------------------

    def __init__(self, sentences: SentenceLike | list[SentenceLike], pooling_method: str = "average", *, normalize: bool = True, chain_index: int | None = None):
        '''
        Arguments
        ---
        sentences: SentenceLike | list[SentenceLike]
            The sentences or chains to include in the chain.
            List should either contain ```Sentence``` or ```SentenceChain``` objects, but never both
        pooling_method: str
            The method used to calculate the chain representative. See ```SentenceChain.VALID_METHODS```. Defaults to ```average```
        normalize: bool
            Normalize the representative after pooling. Defaults to ```True```
        chain_index: int, optional
            The chain's index in the global list of chains for the document
        '''
        self.index = chain_index
        self.parent_cluster = None

        if pooling_method not in SentenceChain.VALID_METHODS:
            raise ValueError(f"Invalid pooling method {pooling_method}")
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

    #----------------------------------------------------------------------------------------------

    @staticmethod
    def pooling_average(sentences: list[SentenceLike], *, normalize: bool = True) -> ndarray:
        vec = np.average(np.row_stack([s.vector for s in sentences]), axis=0)
        return vec / np.linalg.norm(vec) if normalize else vec
    
    @staticmethod
    def pooling_max(sentences: list[SentenceLike], *, normalize: bool = True) -> ndarray:
        vec = np.max(np.row_stack([s.vector for s in sentences]), axis=0)
        return vec / np.linalg.norm(vec) if normalize else vec
    
    @staticmethod
    def pooling_most_similar(sentences: list[SentenceLike], *, normalize: bool = True) -> ndarray:
        sentence_matrix = [sentence.vector for sentence in sentences]
        sims = np.sum(cosine_similarity(sentence_matrix), axis=1)
        most_similar_index = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[0]
        vec = sentences[most_similar_index]
        return vec / np.linalg.norm(vec) if normalize else vec

    @staticmethod
    def pooling(sentences: list[SentenceLike], pooling_method: str, *, normalize: bool = True) -> ndarray:
        '''
        Arguments
        ---
        sentences: list[SentenceLike]
            The sentences to calculate the representative for
        pooling_method: str
            The method used to calculate the representative
        normalize: bool
            Normalize the representative after pooling. Defaults to ```True```

        Returns
        ---
        vec: ndarray
            The representative
        '''

        #But, there was nothing to pool
        if len(sentences) == 1:
            return sentences[0].vector

        match pooling_method:
            case "average": return SentenceChain.pooling_average(sentences, normalize=normalize)
            case "max": return SentenceChain.pooling_max(sentences, normalize=normalize)
            case "most_similar": return SentenceChain.pooling_most_similar(sentences, normalize=normalize)

    #----------------------------------------------------------------------------------------------

    @property
    def vector(self):
        return self._vector
    
    @cached_property
    def text(self):
        return " ".join([s.text for s in self.sentences])
    
    @property
    def doc(self) -> Document:
        return self.sentences[0].doc
    
    @property
    def first_index(self) -> int:
        return self.sentences[0].index
    
    @property
    def last_index(self) -> int:
        return self.sentences[-1].index
    
    @property
    def index_range(self) -> range:
        return range(self.first_index, self.first_index + self.__len__())

    #----------------------------------------------------------------------------------------------
    
    def sentence_matrix(self) -> ndarray:
        '''
        Converts the sentence chain (list) into a matrix, where each row is a sentence. Order is maintained
        '''
        return np.array([x.vector for x in self.sentences])
    
    #----------------------------------------------------------------------------------------------
    
    def get_global(self, global_key: int) -> Sentence:
        '''
        Get a sentence from its index, only if it belongs in this chain's range
        '''
        if global_key not in self.index_range:
            raise ValueError(f"The global sentence index {global_key} is not in the chain's range {self.index_range}")

        return self.sentences[global_key - self.first_index]
    
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
            If set to ```True```, the resulting sentences are always returned as a list, even when 
            we only requested one sentence. Defaults to ```False```

        Returns
        ---
        sentences: Sentence | list[Sentence]
            A list with the next ```n``` sentences. If ```n == 1```, then only one ```Sentence``` is returned,
            unless ```force_list==True```
        '''

        if offset + 1 < self.first_index:
            raise ValueError(f"Chain with offset {self.first_index} cannot provide sentences starting from offset {offset}")

        last_owned_offset = self.first_index + self.__len__() - 1
        max_owned_offset_from_requested = min(offset + n, last_owned_offset)
        remaining_sentences = max(0, offset + n - last_owned_offset)

        #Sentences from the next chain
        extra_sentences = []

        if remaining_sentences > 0:
            #Το 'πα εγώ στον σκύλο μου κι' ο σκύλος στην ουρά του
            extra_sentences = self.next().get_next_sentences(last_owned_offset, remaining_sentences, force_list=True)

        res = self.sentences[(offset + 1 - self.first_index) : (max_owned_offset_from_requested - self.first_index + 1)] + extra_sentences

        if force_list:
            return res
        
        return res if n > 1 else res[0]
    
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
            If set to ```True```, the resulting sentences are always returned as a list, even when 
            we only requested one sentence. Defaults to ```False```

        Returns
        ---
        sentences: Sentence | list[Sentence]
            A list with the previous ```n``` sentences. If ```n == 1```, then only one ```Sentence``` is returned,
            unless ```force_list==True```
        '''
        print(f"get_prev_sentences(requester_offset={offset}, n={n}) len={self.__len__()}")

        if offset + 1 < self.first_index:
            raise ValueError(f"Chain with offset {self.first_index} cannot provide sentences starting from offset {offset}")

        first_owned_offset = self.first_index
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

        print(f"start={min_owned_offset_from_requested - self.first_index}")
        print(f"end={offset - self.first_index}")

        res = extra_sentences + self.sentences[(min_owned_offset_from_requested - self.first_index) : (offset - self.first_index)]

        if force_list:
            return res
        
        return res if n > 1 else res[0]
    
    #--------------------------------------------------------------------------------------------------------------------------

    def next(self, n: int = 1, *, force_list: bool = False) -> Union[SentenceChain, list[SentenceChain]]:
        '''
        Returns the next ```n``` chains from the document where this chain belongs.
        A clustering context must exist for this to work.

        Arguments
        ---
        n: int
            The number of chains to return. Defaults to ```1```
        force_list: bool
            If set to ```True```, the resulting chains are always returned as a list, even when 
            we only requested one chain. Defaults to ```False```

        Returns
        ---
        chains: SentenceChain | list[SentenceChain]
            A list with the next ```n``` chains. If ```n == 1```, then only one ```SentenceChain``` is returned,
            unless ```force_list==True```
        '''
        res = self.parent_cluster.clustering_context.chains[(self.index + 1) : (self.index + 1 + n)]
        if force_list:
            return res
        return res if n > 1 else res[0] 
    
    #--------------------------------------------------------------------------------------------------------------------------
    
    def prev(self, n: int = 1, *, force_list: bool = False) -> Union[SentenceChain, list[SentenceChain]]:
        '''
        Returns the previous ```n``` chains from the document where this chain belongs.
        A clustering context must exist for this to work.

        Arguments
        ---
        n: int
            The number of chains to return. Defaults to ```1```
        force_list: bool
            If set to ```True```, the resulting chains are always returned as a list, even when 
            we only requested one chain. Defaults to ```False```

        Returns
        ---
        chains: SentenceChain | list[SentenceChain]
            A list with the previous ```n``` chains. If ```n == 1```, then only one ```SentenceChain``` is returned,
            unless ```force_list==True```
        '''
        res = self.parent_cluster.clustering_context.chains[self.index - n : self.index]
        if force_list:
            return res
        return res if n > 1 else res[0] 
    
    #--------------------------------------------------------------------------------------------------------------------------

    def __str__(self):
        return f"SentenceChain(index={self.index}, start_offset={self.first_index}, size={self.__len__()}, end_offset={self.first_index + self.__len__() - 1})"
    
    def __repr__(self):
        return f"SentenceChain(i={self.index}, {self.first_index}-{self.first_index + self.__len__() - 1}, size={self.__len__()})"

    def __iter__(self):
        return iter(self.sentences)
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    #For wherever a numpy array is required as input
    def __array__(self, dtype=None):
        return self._vector
    
    def __getitem__(self, key:int) -> Sentence:
        return self.sentences[key]
    
    #--------------------------------------------------------------------------------------------------------------------------

    def data(self) -> dict:
        return {
            'vector': self.vector.tolist(), #Mongo needs list
            'offset': self.first_index,
            'pooling_method': self.pooling_method,
            'index': self.index,
            'sentences': [s.vector.tolist() for s in self.sentences]
        }
    
    @classmethod
    def from_data(cls, data: dict, doc: Document, *, parent: ChainCluster = None) -> SentenceChain:
        obj = cls.__new__(cls)
        obj._vector = np.array(data['vector'])
        obj.pooling_method = data['pooling_method']
        obj.parent_cluster = parent
        obj.index = data.get('index', None)

        #Ensure the document's text has been retrieved
        #Otherwise, we cannot get the sentences
        doc.get()

        offset = data['offset']
        text = split_to_sentences(doc.text, sep="\n")
        if len(text) == 0:
            warnings.warn(f"Document {doc.id} has no sentences")
        obj.sentences = [Sentence(text[offset + i], np.array(vec), doc, offset + i, parent_chain=obj) for i, vec in enumerate(data['sentences'])]

        return obj