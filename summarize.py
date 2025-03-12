
from lxml import etree
from elastic import elasticsearch_client, parse_queries
from helper import Query, Score
import metrics
from transformers import pipeline

client = elasticsearch_client()
collection_path = "collection"
index_name = "test-index"

#===============================================================================================

parse_queries()

'''
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

print(summarizer(summary, max_length=len(summary.split(' '))*1.8, min_length=30, do_sample=False)[0]['summary_text'])'
'''