
from lxml import etree
from query import elasticsearch_client, parse_queries
from helper import Query, Score
import metrics
from transformers import pipeline

client = elasticsearch_client()
collection_path = "collection"
index_name = "test-index"

#===============================================================================================

parse_queries()