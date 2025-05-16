from .classes import SentenceChain
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import sys
from itertools import pairwise
from rich.console import Console
from rich.table import Table

import numpy
#numpy.set_printoptions(threshold=sys.maxsize)

#================================================================================================

def within_chain_similarity(chain: SentenceChain) -> float:
    '''
    Caclculates the average similarity of every pair of sentences in the chain. 

    Args
    ---
    chain: SentenceChain
        The chain to calculate similarity for
    
    Returns
    ---
    sim: float
        Average similarity within the chain
    '''
    if len(chain) == 1:
        return 1
    
    mat = chain.sentence_matrix()
    sim = cosine_similarity(mat, mat)
    res = (np.sum(sim, axis=1) - 1) / (len(chain) - 1)
    return np.average(res)

#================================================================================================

def chain_centroid_similarity(chain: SentenceChain):
    if len(chain) == 1:
        return 1

    mat = chain.sentence_matrix()
    sim = cosine_similarity(mat, chain.vector.reshape((1,-1)))

    if chain.pooling_method in SentenceChain.EXEMPLAR_BASED_METHODS:
        return (np.sum(sim) - 1) / (len(chain) - 1)
    else:
        return np.average(sim)

#================================================================================================

def avg_chain_centroid_similarity(chains: list[SentenceChain], min_size: int = 1, max_size: int|None = None):
    if max_size is None:
        max_size = 6666

    vec = np.array([chain_centroid_similarity(a) for a in chains if len(a) >= min_size and len(a) <= max_size])
    if len(vec) > 0:
        return np.average(vec)
    else:
        return -1


#================================================================================================

def inter_chain_distance(chain_a: SentenceChain, chain_b: SentenceChain):
    dista = cosine_distances(chain_a.sentence_matrix(), chain_b.sentence_matrix())
    res = np.sum(dista, axis=1) / len(chain_b)
    return np.average(res)
    
#================================================================================================

def avg_within_chain_similarity(chains: list[SentenceChain], min_size: int = 1, max_size: int|None = None):
    '''
    Caclculates the average similarity within each chain in the list.
    Then, it calculates the average of those values. 

    Args
    ---
    chains: list[SentenceChain]
        The list of chains to calculate similarity for

    min_size: int
        Default is ```1```. Only consider chains that have at least min_size sentences inside

    max_size: int
        Default is ```None```. Only consider chains that have at most max_size sentences inside

    Returns
    ---
    avg_sim: float
        Average within-chain similarity
    '''
    if max_size is None:
        max_size = 6666

    vec = np.array([within_chain_similarity(a) for a in chains if len(a) >= min_size and len(a) <= max_size])
    if len(vec) > 0:
        return np.average(vec)
    else:
        return -1

#================================================================================================

def min_within_chain_similarity(chain: SentenceChain):
    if len(chain) == 1:
        return 1
    
    mat = chain.sentence_matrix()
    sim = cosine_similarity(mat, mat)
    return np.min(sim)

#================================================================================================

def avg_neighbor_chain_distance(chains: list[SentenceChain]):
    vec = np.array([inter_chain_distance(a, b) for a, b in pairwise(chains)])
    return np.average(vec)

#================================================================================================

def avg_chain_length(chains: list[SentenceChain]):
    return np.average(np.array([len(c) for c in chains]))



#================================================================================================

def chain_metrics(chains: list[SentenceChain]):
    console = Console()
    table = Table()

    table.add_column("Metric")
    table.add_column("Score")

    chain_centroid_similarity(chains[0])

    table.add_row("Average Within-Chain Similarity", str(np.round(avg_within_chain_similarity(chains), decimals=3)))
    table.add_row("Average Within-Chain Similarity (len >= 2)", f"{np.round(avg_within_chain_similarity(chains, min_size=2), decimals=3):.3f}")
    table.add_row("Average Within-Chain Similarity (len >= 3)", f"{np.round(avg_within_chain_similarity(chains, min_size=3), decimals=3):.3f}")
    table.add_row("Average Within-Chain Similarity (len >= 4)", f"{np.round(avg_within_chain_similarity(chains, min_size=4), decimals=3):.3f}")
    table.add_row("Average Within-Chain Similarity (len >= 5)", f"{np.round(avg_within_chain_similarity(chains, min_size=5), decimals=3):.3f}")
    table.add_row("Average Within-Chain Similarity (len >= 6)", f"{np.round(avg_within_chain_similarity(chains, min_size=6), decimals=3):.3f}")
    table.add_row("Average Within-Chain Similarity (len = 4)", f"{np.round(avg_within_chain_similarity(chains, min_size=4, max_size=4), decimals=3):.3f}")
    table.add_row("Average Neighbor Chain Distance", f"{np.round(avg_neighbor_chain_distance(chains), decimals=3):.3f}")
    table.add_row("Average Chain Length", f"{np.round(avg_chain_length(chains), decimals=3):.3f}")
    table.add_row("Global Minimum Within-Chain Similarity", f"{np.round(np.min(np.array([min_within_chain_similarity(c) for c in chains]))):.3f}")

    console.print(table)