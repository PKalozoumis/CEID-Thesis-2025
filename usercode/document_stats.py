#Some documents are generating too many clusters, while others are just fine
#For example, I'm using this list:
#[1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 6415]
#Documents 4355, and 372 have too many clusters and many outliers
#What makes those documents different?

import os
import sys
sys.path.append(os.path.abspath(".."))

from collections import namedtuple
from mypackage.elastic import Session, ElasticDocument

import numpy as np

ProcessedDocument = namedtuple("ProcessedDocument", ["chains", "labels", "clusters"])

import pickle
from rich.console import Console
from itertools import chain

console = Console()

docs_path = "../notebooks/pipeline/pickles"
docs_list = [1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 6415]

pkl = []

for fname in map(lambda x: f"{docs_path}/{x}.pkl", docs_list):
    with open(fname, "rb") as f:
        pkl.append(pickle.load(f))


for i, p in enumerate(pkl):
    data = {'id':docs_list[i], 'index': i}

    chain_lengths = [len(c) for c in p.chains]
    data['num_chains'] = len(p.chains)
    data['avg_chain_length'] = np.average(chain_lengths)
    data['min_chain_length'] = np.min(chain_lengths)
    data['max_chain_length'] = np.max(chain_lengths)

    sentence_lengths = [len(c) for c in chain.from_iterable(p.chains)]
    data['num_sentences'] = np.sum(sentence_lengths)
    data['avg_sentence_length'] = np.average(sentence_lengths)
    data['min_sentence_length'] = np.min(sentence_lengths)
    data['max_sentence_length'] = np.max(sentence_lengths)

    console.print(data)

print(len(pkl))



