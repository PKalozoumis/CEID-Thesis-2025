import os
import sys
sys.path.append(os.path.abspath(".."))

from mypackage.elastic import Session, ElasticDocument
from rich.console import Console


console = Console()

#===============================================================================================================

if __name__ == "__main__":
    sess = Session("pubmed-index", base_path="..")
    doc = ElasticDocument(sess, 1923, cache_dir="cache", text_path="summary")

    console.print(doc.text)