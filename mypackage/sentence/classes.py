from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from mypackage.elastic import Document, ElasticDocument
from functools import cached_property
import json
from itertools import chain

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
    
#============================================================================================
    
class SentenceChain(SentenceLike):
    '''
    Represents a chain of one or more consecutive sentences that are very similar
    '''
    _vector: ndarray
    sentences: list[SentenceLike]

    def __init__(self, sentences: list[SentenceLike], pooling_method: str = "average"):
        self._vector = SentenceChain.pooling(sentences, pooling_method)

        if isinstance(sentences[0], Sentence):
            self.sentences = sentences
        elif isinstance(sentences[0], SentenceChain):
            #Iterate over the sentences in the chains
            #Combine them all (in order) as part of the new chain
            self.sentences = list(chain.from_iterable(sentences))

    @staticmethod
    def pooling_average(sentences: list[Sentence]) -> ndarray:
        return np.average(np.row_stack([s.vector for s in sentences]), axis=0)
    
    @staticmethod
    def pooling_max(sentences: list[Sentence]) -> ndarray:
        return np.max(np.row_stack([s.vector for s in sentences]), axis=0)

    @staticmethod
    def pooling(sentences: list[Sentence], pooling_method: str) -> ndarray:
        match pooling_method:
            case "average": return SentenceChain.pooling_average(sentences)
            case "max": return SentenceChain.pooling_max(sentences)
    
    def sentence_matrix(self) -> ndarray:
        '''
        Converts the sentence chain (list) into a matrix, where each row is a sentence. Order is maintained
        '''
        return np.array([x.vector for x in self.sentences])

    def __iter__(self):
        return iter(self.sentences)
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __array__(self, dtype=None):
        return self._vector
    
    def __getitem__(self, key:int):
        return self.sentences[key]
    
    def __str__(self):
        return json.dumps([s.text for s in self.sentences])
    
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