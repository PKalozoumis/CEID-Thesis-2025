
from lxml import etree
from helper import Score, Query, elasticsearch_client
import metrics
from query import parse_queries
from transformers import pipeline

client = elasticsearch_client()
collection_path = "collection"
index_name = "test-index"

#===============================================================================================

if __name__ == "__main__":

    