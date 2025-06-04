
from __future__ import annotations

from sentence_transformers import SentenceTransformer
from itertools import pairwise, starmap
import re

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

def doc_to_sentences(doc: Document, transformer: SentenceTransformer, *, remove_duplicates: bool = False, remove_empty: bool =True, sep: str | None = "\n") -> list[Sentence]:
    '''
    Breaks down a document into sentences. For the entire set of sentences, the embeddings are calculated

    Arguments
    ---
    doc: Document
        The document to extract sentences from
    transformer: SentenceTransformer
        The model that will generate the embeddings
    remove_duplicates: bool
        Removes duplicate sentences (my dataset was goofy, and I've already indexed and cached the docs)
    remove_empty: bool
        Removes empty sentences. Defaults to ```True```
    sep: str
        The delimiter for splitting the document into sentences. Defaults to newline
        
    Returns
    ---
    out: list[Sentence]
        A list of the document's sentences as ```Sentence``` objects
    '''
    sentences = split_to_sentences(doc.text, sep=sep)

    if remove_duplicates:
        #Deduplicate sentences
        #Some documents accidentally have the same sentences multiple times BECAUSE THE CREATORS ARE CLOWNS 😬😬😬
        #Goofy ahh dataset bro istg

        #Btw, ideally this should happen during indexing, but I've already cached these docs...

        seen_sentences = set()
        deduplicated = []

        for sentence in sentences:
            if len(sentence.split()) > 7:
                if sentence in seen_sentences:
                    #console.print(f"{doc.id}: {sentence}")
                    print("IMPOSTOR DETECTED 🗣")
                    pass
                else:
                    deduplicated.append(sentence)
                    seen_sentences.add(sentence)
    else:
        deduplicated = sentences

    if remove_empty:
        deduplicated = [txt for txt in deduplicated if not re.match(r"^\s*$", txt)]

    embeddings = transformer.encode(deduplicated)

    result = []
    result: list[Sentence]
    for offset, (sentence, embedding) in enumerate(zip(deduplicated, embeddings)):
        result.append(Sentence(sentence, embedding, doc, offset))

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
    pairs = [SimilarityPair.from_sentences(s1, s2) for s1, s2 in pairwise(sentences)]

    #No more merging can happen, since all pairs are below the threshold
    #-------------------------------------------------------------------------------------
    if not any(filter(lambda x: x.sim >= threshold, pairs)):
        #Very rare. Failure on the first round
        if isinstance(sentences[0], Sentence):
            print("If you're reading this, there might be a problem")
            return [SentenceChain(s, index=i) for i, s in enumerate(sentences)] 
        
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

    result = [SentenceChain(c, pooling_method, normalize=normalize) for c in chains]
    
    if round_limit is None:
        return iterative_merge(result, threshold=threshold, round_limit=None, pooling_method=pooling_method, normalize=normalize)
    elif round_limit > 1:
        return iterative_merge(result, threshold=threshold, round_limit=round_limit-1, pooling_method=pooling_method, normalize=normalize)
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