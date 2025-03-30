from classes import SentenceChain
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import sys

import numpy
numpy.set_printoptions(threshold=sys.maxsize)

def intra_chain_distance(chain: SentenceChain):
    cosine_distances(chain.sentence_matrix(), chain.sentence_matrix())

#def chain_silhouette_score():
