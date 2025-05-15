import os
import sys
sys.path.append(os.path.abspath(".."))

from mypackage.elastic import Session, Query
from rich.console import Console


console = Console()

#===============================================================================================================

if __name__ == "__main__":
    sess = Session("pubmed-index", base_path="..")
    query = Query(0, "What are the primary behaviours and lifestyle factors that contribute to childhood obesity", source=["summary", "article"], text_path="article", cache_dir="cache")
    res = query.execute(sess)
    console.print(res)

'''
[
    ElasticDocument(id=1923),
    ElasticDocument(id=4355),
    ElasticDocument(id=4166),
    ElasticDocument(id=3611),
    ElasticDocument(id=6389),
    ElasticDocument(id=272),
    ElasticDocument(id=2635),
    ElasticDocument(id=2581),
    ElasticDocument(id=372),
    ElasticDocument(id=6415)
]
'''