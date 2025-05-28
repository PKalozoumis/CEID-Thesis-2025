#from mypackage.query_generator import generate_query
from mypackage.helper.llm import generate_query
from mypackage import ElasticDocument, Session
import os
from sentence_transformers import SentenceTransformer
import json


out_file = open("queries.txt", "w")

'''
for i in range(30):
    file = f"cached_docs/pubmed_{i:04}.json"
    doc = Document.from_json(file, text_path="summary")
    print(f"For document {file}\n{'='*30}")
    res = generate_query(doc.text)
    print(res, end="\n\n")
    out_file.write(json.dumps({'doc': i, **res}) + "\n")

'''
sess = Session("pubmed-index")
docs = (ElasticDocument(sess, i, text_path="summary") for i in range(113,6000))

for doc in docs:
    print(f"For document {doc.id:04}\n{'='*30}")
    res = generate_query(doc.text)
    print(res, end="\n\n")
    out_file.write(json.dumps({'doc': doc.id, **res}) + "\n")

out_file.close()