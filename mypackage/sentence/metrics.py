from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from itertools import pairwise

from rich.console import Console
from rich.table import Table

from ..helper import create_table
from .classes import SentenceChain


def chaining_ratio(chains: list[SentenceChain]) -> float:
    count = 0
    total = 0
    for c in chains:
        total += len(c)
        if len(c) > 1:
            count += len(c)

    return count/total

#================================================================================================

def within_chain_similarity(chain: SentenceChain) -> float:
    '''
    Caclculates the average similarity of every pair of sentences in the chain. 

    Arguments
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

def chain_centroid_similarity(chain: SentenceChain, *, allow_self_similarity: bool = False):
    '''
    Calculates the average similarity between every sentence in the chain and the chain representative
    '''
    if len(chain) == 1:
        return 1.0

    mat = chain.sentence_matrix()
    sim = cosine_similarity(mat, chain.vector.reshape((1,-1)))

    if not allow_self_similarity and chain.pooling_method in SentenceChain.EXEMPLAR_BASED_METHODS:
        return (np.sum(sim) - 1) / (len(chain) - 1)
    else:
        return np.average(sim)

#================================================================================================

def avg_chain_centroid_similarity(chains: list[SentenceChain], min_size: int = 1, max_size: int|None = None, vector=False, *, allow_self_similarity: bool = False):
    '''
    For each chain in the list, it calculates the average similarity between every sentence in the chain and the chain representative
    Then, it calculates the average of those values. 

    Arguments
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
        Average centroid similarity
    '''
    if max_size is None:
        max_size = 6666

    vec = np.array([chain_centroid_similarity(a, allow_self_similarity=allow_self_similarity) for a in chains if len(a) >= min_size and len(a) <= max_size])

    if vector:
        return vec.tolist()
    else:
        if len(vec) > 0:
            return np.average(vec)
        else:
            return np.nan

#================================================================================================

def inter_chain_distance(chain_a: SentenceChain, chain_b: SentenceChain):
    '''
    Calculates the average distance between every sentence of one chain and every sentence of another chain
    '''
    dista = cosine_distances(chain_a.sentence_matrix(), chain_b.sentence_matrix())
    res = np.sum(dista, axis=1) / len(chain_b)
    return np.average(res)
    
#================================================================================================

def avg_within_chain_similarity(chains: list[SentenceChain], min_size: int = 1, max_size: int|None = None, size_index: dict[int, list[SentenceChain]] = None):
    '''
    Caclculates the average similarity within each chain in the list.
    Then, it calculates the average of those values. 

    Arguments
    ---
    chains: list[SentenceChain]
        The list of chains to calculate similarity for
    min_size: int
        Default is ```1```. Only consider chains that have at least min_size sentences inside
    max_size: int
        Default is ```None```. Only consider chains that have at most max_size sentences inside
    size_index: dict[int, list[SentenceChain]]
        A precalculated dictionary mapping each size to the chains of that size
        
    Returns
    ---
    avg_sim: float
        Average within-chain similarity
    '''
    if max_size is None:
        max_size = 6666

    if size_index:
        vec = np.array([within_chain_similarity(c) for k in range(min_size, max_size+1) if k in size_index for c in size_index[k]])
    else:
        vec = np.array([within_chain_similarity(a) for a in chains if len(a) >= min_size and len(a) <= max_size])
    
    if len(vec) > 0:
        return np.average(vec)
    else:
        return np.nan

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

def chain_metrics(chains: list[SentenceChain], *, render=False, return_renderable=False) -> dict | tuple[dict, Table]:

    metrics = {
        'avg_sim': {'name': "Average Within-Chain Similarity", 'value': avg_within_chain_similarity(chains)},
        'avg_sim_2': {'name': "Average Within-Chain Similarity (len >= 2)", 'value': avg_within_chain_similarity(chains, min_size=2)},
        'avg_sim_3': {'name': "Average Within-Chain Similarity (len >= 3)", 'value': avg_within_chain_similarity(chains, min_size=3)},
        'avg_sim_4': {'name': "Average Within-Chain Similarity (len >= 4)", 'value': avg_within_chain_similarity(chains, min_size=4)},
        'avg_sim_5': {'name': "Average Within-Chain Similarity (len >= 5)", 'value': avg_within_chain_similarity(chains, min_size=5)},
        'avg_sim_6': {'name': "Average Within-Chain Similarity (len >= 6)", 'value': avg_within_chain_similarity(chains, min_size=6)},
        'avg_sim_eq4': {'name': "Average Within-Chain Similarity (len = 4)", 'value': avg_within_chain_similarity(chains, min_size=4, max_size=4)},
        'avg_dist': {'name': "Average Neighbor Chain Distance", 'value': avg_neighbor_chain_distance(chains)},
        'avg_len': {'name': "Average Chain Length", 'value': avg_chain_length(chains)},
        #'min_sim': {'name': "Global Minimum Within-Chain Similarity", 'value': np.min(np.array([min_within_chain_similarity(c) for c in chains]))},
        'avg_centroid_sim': {'name': "Average Similarity to Centroid", 'value': avg_chain_centroid_similarity(chains)},
        'avg_centroid_sim_2': {'name': "Average Similarity to Centroid (len >= 2)", 'value': avg_chain_centroid_similarity(chains, min_size=2)},
        'avg_centroid_sim_6': {'name': "Average Similarity to Centroid (len >= 6)", 'value': avg_chain_centroid_similarity(chains, min_size=6)}
    }

    if render or return_renderable:
        table = create_table(['Metric', 'Score'], {temp['name']:temp['value'] for temp in metrics.values()}, title="Chaining Metrics")
        if render:
            console = Console()
            console.print(table)
        if return_renderable:
            return metrics, table 
    
    return metrics