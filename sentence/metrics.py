from classes import SentenceChain
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import sys
from itertools import pairwise

import numpy
#numpy.set_printoptions(threshold=sys.maxsize)

#================================================================================================

def intra_chain_distance(chain_a: SentenceChain, *, vec_size: int = None):
    if vec_size is None:
        vec_size = len(chain_a)

    if len(chain_a) == 1:
        return np.zeros((vec_size,))
    
    mat = chain_a.sentence_matrix()
    dista = cosine_distances(mat, mat)
    res = np.sum(dista, axis=1) / (len(chain_a) - 1)
    return np.pad(res, (0, vec_size - len(res)), mode="constant", constant_values=0)

#================================================================================================

def inter_chain_distance(chain_a: SentenceChain, chain_b: SentenceChain, *, vec_size: int = None):
    if vec_size is None:
        vec_size = len(chain_a)
    
    dista = cosine_distances(chain_a.sentence_matrix(), chain_b.sentence_matrix())
    res = np.sum(dista, axis=1) / len(chain_b)
    return np.pad(res, (0, vec_size - len(res)), mode="constant", constant_values=0)
    
#================================================================================================

#Failure
def chain_silhouette_score(chains: list[SentenceChain]):

    max_len = max(map(len, chains))

    #Each row represents one pair
    vec_a = np.array([intra_chain_distance(a, vec_size=max_len) for a in chains])
    vec_b = np.array([inter_chain_distance(a, b, vec_size=max_len) for a, b in pairwise(chains)] + [[0 for _ in range(max_len)]])
    max_vec = np.max(np.stack([vec_a, vec_b], axis=0), axis=0)

    print(vec_a)
    print(vec_b)
    print(max_vec)

    res = np.nan_to_num((vec_b - vec_a)/max_vec)
    print(res)



    res = np.sum(vec_b, axis=1)
    print(res)
