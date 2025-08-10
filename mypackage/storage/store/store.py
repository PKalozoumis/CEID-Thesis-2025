from __future__ import annotations

import pickle
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...clustering import ChainClustering

#==========================================================================================================

def save_clusters(clustering: ChainClustering, path: str, *, params: dict = None):
    '''
    Saves clusters of one specific document to a pickle file

    Arguments
    ---
    clusters: dict
        The clusters returned by chain_clustering
    path: str
        Path to store the pickle files in
    params: dict, optional
        The parameters used for all operations (e.g. chaining threshold, pooling methods, UMAP parameters, etc)
    '''
    out = clustering.data()

    with open(os.path.join(path, f"{out[0]['id']}.pkl"), "wb") as f:
        pickle.dump({'params': params, 'data': out}, f)

#=====================================================================================================

