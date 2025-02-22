
from helper import Score, Query, elasticsearch_client, parse_queries
import metrics
from transformers import pipeline

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

client = elasticsearch_client()
collection_path = "collection"
index_name = "test-index"

#===============================================================================================

if __name__ == "__main__":

    queries = parse_queries()[98:99] #99 queries

    multiple_query_results = []
    docs = []

    for query in queries:
        search_body = {
            "match": {"abstract": query.text}
        }

        res = client.search(index=index_name, query=search_body, filter_path=["hits.hits._source.abstract", "hits.hits._id"])['hits']
        docs.append([temp['_source']['abstract'] for temp in res['hits']])
        multiple_query_results.append([int(temp['_id']) for temp in res['hits']])
    
    print(f"Average NDCG: {metrics.average_ndcg(multiple_query_results, queries)[-1]}")

    hyperdoc = ". ".join(docs[0])
    #print(hyperdoc)

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cpu")

    summary  = ""

    #1024 tokens
    for doc in docs[0]:
        print(f"Original\n{'-'*40}\n{doc}\n")
        temp = summarizer(doc, max_length=len(doc.split(' '))*1.2, min_length=30, do_sample=False)[0]['summary_text']
        print(f"Summary\n{'-'*40}\n{temp}\n")
        print("="*80 + "\n")
        summary += temp + ". "

    print("FINAL RESULT\n\n")

    print(summary)

    print("\n\n")

    print(summarizer(summary, max_length=len(summary.split(' '))*1.8, min_length=30, do_sample=False)[0]['summary_text'])