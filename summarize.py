
from lxml import etree

from helper import Score, Query, elasticsearch_client
import metrics
from query import parse_queries

client = elasticsearch_client()
collection_path = "collection"
index_name = "test-index"

#===============================================================================================

if __name__ == "__main__":

    queries = parse_queries() #99 queries

    multiple_query_results = []

    for query in queries:
        search_body = {
            "match": {"abstract": query.text}
        }

        res = client.search(index=index_name, query=search_body, filter_path=["hits.hits._id"])["hits"]["hits"]
        res = [int(id) for temp in res for id in temp.values()]
        multiple_query_results.append(res)

    print(f"Average NDCG: {metrics.average_ndcg(multiple_query_results, queries)[-1]}")