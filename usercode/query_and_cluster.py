import os
import sys
sys.path.append(os.path.abspath(".."))

from mypackage.elastic import Session, ElasticDocument
from mypackage.clustering import chain_clustering
from rich.console import Console
from functools import partial


console = Console()

#===============================================================================================================

if __name__ == "__main__":
    #sess = Session("pubmed-index", base_path="..")
    #doc = ElasticDocument(None, 1923, cache_dir="cache", text_path="summary")

    docs_to_retrieve = [1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 6415]
    docs = list(map(partial(ElasticDocument, None, text_path="article", cache_dir = "cache"), docs_to_retrieve))
    console.print(docs)