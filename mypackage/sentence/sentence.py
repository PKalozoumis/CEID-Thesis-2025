from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from kmedoids import KMedoids
from kneed import KneeLocator
from matplotlib import pyplot as plt
from ..elastic import elasticsearch_client, ScrollingCorpus, Session, Document
from itertools import pairwise, starmap
from ..helper import panel_print, lock_kwargs
from .helper import split_to_sentences
from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown
from .metrics import avg_neighbor_chain_distance, avg_within_chain_similarity, chain_metrics
from .classes import Sentence, SentenceChain, SentenceLike, SimilarityPair
from functools import partial, wraps

console = Console()

#============================================================================================

def doc_to_sentences(doc: Document, transformer: SentenceTransformer, remove_duplicates = False) -> list[Sentence]:
    '''
    Breaks down a document into sentences. For the entire set of sentences, the embeddings are calculated

    Arguments
    ---
    doc: Document
        The document to extract sentences from
    transformer: SentenceTransformer
        The model that will generate the embeddings
    remove_duplicates: bool
        Removes sentences that are 

    Returns
    ---
    out: list[Sentence]
        A list of the document's sentences as ```Sentence``` objects
    '''
    sentences = split_to_sentences(doc.text)

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
                    pass
                else:
                    deduplicated.append(sentence)
                    seen_sentences.add(sentence)
    else:
        deduplicated = sentences

    embeddings = transformer.encode(deduplicated)

    result = []
    for offset, (sentence, embedding) in enumerate(zip(deduplicated, embeddings)):
        result.append(Sentence(sentence, embedding, doc, offset))

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

def buggy_merge(sentences: list[SentenceLike],*, threshold: float, round_limit: int | None = 1, pooling_method="average"):
    '''
    Clusters a list of sentence chains for a single document.
    The chains inside each returned cluster are ordered based on their offset inside the document

    Arguments
    --------------------------------------------------------
    chain: list[SentenceChain]
        The list of chains to cluster

    n_components: int
        The number of dimensions to reduce the embedding space to.
        Set to ```None``` to skip dimensionality reduction

    Returns
    --------------------------------------------------------
    labels: list[int]
        A list of labels. One label for each input chain

    clustered_chains: dict[int, ChainCluster]
        A dictionary of clusters, with the label as the key
    '''
    pairs = [SimilarityPair.from_sentences(s1, s2) for s1, s2 in pairwise(sentences)]

    #No more merging can happen
    if not any(filter(lambda x: x.sim > threshold, pairs)):
        return sentences

    chains = []

    for i, pair in enumerate(pairs):

        if i == 0:
            chains.append([pair.s1, pair.s2])
            continue
        
        if pair.sim >= threshold: #Add to the chain
            chains[-1].append(pair.s2)
        else: #Create new chain for this sentence
            chains.append([pair.s2])

    result = [SentenceChain(c, pooling_method) for c in chains]
    
    if round_limit is None:
        return buggy_merge(result, threshold=threshold, round_limit=None, pooling_method=pooling_method)
    elif round_limit > 1:
        return buggy_merge(result, threshold=threshold, round_limit=round_limit-1, pooling_method=pooling_method)
    else:
        return result
    
#====================================================================================================================

def iterative_merge(sentences: list[SentenceLike],*, threshold: float, round_limit: int | None = 1, pooling_method: str = "average") -> list[SentenceLike]:
    '''
    Clusters a list of sentence chains for a single document.
    The chains inside each returned cluster are ordered based on their offset inside the document

    Arguments
    --------------------------------------------------------
    sentences: list[SentenceLike]
        The list of sentences to chain

    threshold: float
        The cosine similarity threshold for two ```SentenceLike``` objects to be considered similar enough
        to merge into the same chain. Takes values between ```0``` and ```1```
        
    round_limit: int | None
        The number of rounds. On the first round we try to chain as many sentences from the document as possible, using
        the merging method (here ```iterative_merge```). After chaining, it's possible that the chains can
        also be chained further, depending on what their new vector is (affected by ```pooling_method```, so this can
        continue for more rounds. We can set this max number of rounds, which by default is set to ```1```.
        Setting to ```None``` removes the limit entirely. Setting to ```0``` performes no chaining and just returns
        the original sentences

    pooling_method: str
        The method we use to generate the new chain embedding from the partial ```SentenceLike``` objects' embeddings.
        By default it's ```average```

    Returns
    --------------------------------------------------------
    chains: list[SentenceLike]
        A list of chains. If ```round_limit == 0```, then the original set of sentences is returned
    '''
    #Disable chaining
    if round_limit == 0:
        return list(map(SentenceChain, sentences))
    
    #We check the sentences in pairs to see if their similarity is above the threshold
    pairs = [SimilarityPair.from_sentences(s1, s2) for s1, s2 in pairwise(sentences)]

    #No more merging can happen, sicne all pairs are below the threshold
    if not any(filter(lambda x: x.sim >= threshold, pairs)):
        return sentences

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

    result = [SentenceChain(c, pooling_method) for c in chains]
    
    if round_limit is None:
        return iterative_merge(result, threshold=threshold, round_limit=None, pooling_method=pooling_method)
    elif round_limit > 1:
        return iterative_merge(result, threshold=threshold, round_limit=round_limit-1, pooling_method=pooling_method)
    else:
        return result

#============================================================================================

def chaining(method: str):

    match method:
        case 'iterative': return iterative_merge
        case 'buggy': return buggy_merge
        case 'none': return lock_kwargs(iterative_merge, round_limit=0)
        case _: return None

#============================================================================================

def sentence_clustering(embeddings):
    '''
    Returns:
    - Labels
    - Medoids
    '''
    dista = cosine_distances(embeddings)

    inertia = []
    K_range = list(range(1, len(embeddings)))

    #Find optimal cluster count
    for k in K_range:
        clustering = KMedoids(n_clusters=k, metric="precomputed")
        clustering_model = clustering.fit(dista)
        inertia.append(clustering_model.inertia_)
        print(clustering_model.inertia_)

    knee_locator = KneeLocator(K_range, inertia, curve="convex", direction="decreasing")
    optimal_k = knee_locator.elbow
    optimal_k = 3
    print(optimal_k)

    fig, ax = plt.subplots()
    ax.plot(K_range, inertia, "ro--")
    #plt.show()

    #Cluster optimal
    clustering = KMedoids(n_clusters=int(optimal_k), metric="precomputed")
    clustering_model = clustering.fit(dista)
    medoids = clustering_model.medoid_indices_
    print(f"Clustering: {clustering_model.labels_}")
    print(f"Medoids: {medoids}")

    return clustering_model.labels_, medoids

    #sorted_sentences = sorted(sentences, key=lambda x: query_sim[sentences.index(x)], reverse=True)
    #print(sorted_sentences)

#============================================================================================

if __name__ == "__main__":
    session = Session(elasticsearch_client("../credentials.json", "../http_ca.crt"), "arxiv-index")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    corpus = ScrollingCorpus(session, batch_size=10, doc_field="article")

    for doc in corpus:
        sentences = doc_to_sentences(doc, model)
        merged = iterative_merge(sentences, threshold=0.6, round_limit=None, pooling_method="average")
        print(len(merged[0]))
        print(merged[0].sentences)
        panel_print(merged[0].text)
        
        chain_metrics(merged)

        '''
        labels, medoids = sentence_clustering(embeddings)
        sorted_data = sorted(zip(labels, sentences), key=lambda x: x[0])
        clusters = {k: [v for _, v in g] for k, g in groupby(sorted_data, key=lambda x: x[0])}
        print(clusters)'
        '''
        break