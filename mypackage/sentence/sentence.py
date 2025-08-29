
from __future__ import annotations

from sentence_transformers import SentenceTransformer
from itertools import pairwise, starmap
import re
from nltk.tokenize import sent_tokenize
import numpy as np
import warnings

from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown

from ..helper import lock_kwargs
from .helper import split_to_sentences
from .classes import Sentence, SentenceChain, SentenceLike, SimilarityPair

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..elastic import Document

console = Console()

#============================================================================================

def doc_to_sentences(doc: Document, transformer: SentenceTransformer | None, *, remove_empty: bool = True, sep: str | None = None, precalculated_embeddings: list = None) -> list[Sentence]:
    '''
    Breaks down a document into sentences. For the entire set of sentences, the embeddings are calculated

    Arguments
    ---
    doc: Document
        The document to extract sentences from
    transformer: SentenceTransformer
        The model that will generate the embeddings. Set ```None``` to break the document into sentences without calculating embeddings
    remove_empty: bool
        Removes empty sentences. Defaults to ```True```
    sep: str | None
        The separator used for splitting. Defaults to ```None```, meaning that ```nltk.tokenize.sent_tokenize``` is
        used to automatically detect sentence boundaries
        
    Returns
    ---
    out: list[Sentence]
        A list of the document's sentences as ```Sentence``` objects
    '''
    sentences = split_to_sentences(doc.text, sep=sep)

    if remove_empty:
        sentences = [txt for txt in sentences if not re.match(r"^\s*$", txt)]

    result = []

    if precalculated_embeddings is not None or transformer is not None:
            
        if precalculated_embeddings is not None:
            if len(precalculated_embeddings) != len(sentences):
                raise Exception(f"Number of embeddings ({len(precalculated_embeddings)}) is different from number of sentences ({len(sentences)})")
            embeddings = precalculated_embeddings
        else: #meaning that precalculated_embeddings was None, and transformer was NOT none
            embeddings = transformer.encode(sentences)

        for offset, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            result.append(Sentence(sentence, embedding, doc, offset))
    else:
        for offset, sentence in enumerate(sentences):
            result.append(Sentence(sentence, None, doc, offset))

    doc.sentences = result
    return result

#============================================================================================

def print_pairs(sentences):
    console.clear()

    table = Table()
    table.add_column("Sentence Pair")
    table.add_column("Similarity", vertical="top")

    for thing in starmap(lambda x, y: (x,y,x.similarity(y)), pairwise(sentences)):
        mytext = f'''
- {thing[0].text.strip()}


- {thing[1].text.strip()}

---
        '''
        table.add_row(Markdown(mytext), "\n\n"+str(thing[2]))

    console.print(table)

#============================================================================================

def iterative_merge(
        sentences: list[SentenceLike],
        *,
        threshold: float = 0.6,
        round_limit: int | None = 1,
        pooling_method: str = "average",
        min_chains: int = 28,
        normalize: bool = True
    ) -> list[SentenceChain]:
    '''
    Chains together a list of sentence chains for a single document.
    The sentences inside each returned chain are ordered based on their offset inside the document

    Arguments
    ---
    sentences: list[SentenceLike]
        The list of sentences to chain

    threshold: float
        The cosine similarity threshold for two ```SentenceLike``` objects to be considered similar enough
        to merge into the same chain. Takes values between ```0``` and ```1```. Defaults to ```0.6```
        
    round_limit: int | None
        The number of rounds. On the first round we try to chain as many sentences from the document as possible, using
        the merging method (here ```iterative_merge```). After chaining, it's possible that the chains can
        also be chained further, depending on what their new vector is (affected by ```pooling_method```, so this can
        continue for more rounds. We can set this max number of rounds, which by default is set to ```1```.
        Setting to ```None``` removes the limit entirely. Setting to ```0``` performes no chaining and just returns
        the original sentences

    pooling_method: str
        The method we use to generate the new chain vector from the partial ```SentenceLike``` objects' embeddings.
        By default it's ```average```

    min_chains: int
        Lower bound for the number of chains. If during at iteration the number of chains drops below the min, we fallback to the
        previous round's results

    normalize: bool
        Normalize the representative after pooling. Defaults to ```True```

    Returns
    ---
    chains: list[SentenceChain]
        A list of chains. If ```round_limit == 0```, then the original set of sentences is returned, as single-sentence chains
    '''
    #round_limit will only reach down to 1 on normal operation (recursion)
    #This below will never happen unless you explicitly set the parameter to 0
    #So 0 essentially disables chaining
    #This will create a separate chain for each sentence
    if round_limit == 0:
        #Nothing is affected. No pooling happens
        return [SentenceChain(s, index=i) for i, s in enumerate(sentences)] 
    
    #We check the sentences in pairs to see if their similarity is above the threshold
    #Here, it doesn't matter if ```sentences``` is a list of sentences or chains
    #Either way, the SimilarityPair uses the ```vector``` property which they both have
    #This means that starting from the second round, we compare chains the same way we compared sentence in the first round
    #...but using the representative as the vector instead
    pairs = [SimilarityPair.from_sentence_like(s1, s2) for s1, s2 in pairwise(sentences)]

    #No more merging can happen, since all pairs are below the threshold
    #-------------------------------------------------------------------------------------
    if not any(filter(lambda x: x.sim >= threshold, pairs)):
        #Very rare. Failure on the first round, because no pair is above the threshold
        if isinstance(sentences[0], Sentence):
            warnings.warn("No pair of sentences was above the similarity threshold in the first round")
            return [SentenceChain(s, chain_index=i) for i, s in enumerate(sentences)] 
        
        for i,s in enumerate(sentences):
            s.index = i
        return sentences
    
    #-------------------------------------------------------------------------------------

    chains: list[list[SentenceLike]] = []
    appended_first_full_pair = False

    for i, pair in enumerate(pairs):
        #While we still haven't appended a full pair,
        #then we exclusively create new chains, each with one sentence
        if not appended_first_full_pair:
            if pair.sim >= threshold:
                chains.append([pair.s1, pair.s2])
                appended_first_full_pair = True
            else:
                chains.append([pair.s1])
        #After adding the first full pair, we only need to
        #add the second element of any subsequent pair, because they're overlapping
        else:
            if pair.sim >= threshold: #Add to the chain
                chains[-1].append(pair.s2)
            else: #Create new chain for this sentence
                chains.append([pair.s2])

    #-------------------------------------------------------------------------------------

    if len(chains) < min_chains:
        warnings.warn(f"Created {len(chains)} chains, which is below the minimum threshold of {min_chains}. Using {len(sentences)} chains of the previous round")
        if isinstance(sentences[0], Sentence):
            #Failure on the first round. Sentences must become chains
            return [SentenceChain(s, chain_index=i) for i, s in enumerate(sentences)] 
        return sentences
    else:
        result = [SentenceChain(c, pooling_method, normalize=normalize) for c in chains]
    
        if round_limit is None:
            return iterative_merge(result, threshold=threshold, round_limit=None, pooling_method=pooling_method, normalize=normalize, min_chains=min_chains)
        elif round_limit > 1:
            return iterative_merge(result, threshold=threshold, round_limit=round_limit-1, pooling_method=pooling_method, normalize=normalize, min_chains=min_chains)
        else: #round_limit == 1
            for i,s in enumerate(result):
                s.index = i
            return result

#============================================================================================

def chaining(method: str):
    match method:
        case 'iterative': return iterative_merge
        case 'none': return lock_kwargs(iterative_merge, round_limit=0)
        case _: return None