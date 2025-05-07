from .classes import Session, Query, ElasticDocument

#===============================================================================================================

if __name__ == "__main__":
    from rich.console import Console
    console = Console()

    sess = Session("pubmed-index")
    query = Query(0, "Parkinson's disease cognitive impairment symptoms", match_field="article", source=["article_id", "summary"], text_path="article_id")
    res = query.execute(sess)
    console.print(res)
    console.print(res[0].text)