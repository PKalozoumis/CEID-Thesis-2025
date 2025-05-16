#Let's see if the text in the retrieved document is the same as the text from the pickled document

import sys
import os
sys.path.append(os.path.abspath("../.."))

from mypackage.elastic import Session, ElasticDocument
from mypackage.storage import load_pickles

sess = Session("arxiv-index", cache_dir="../cache", use="cache")

for i in range(10):
    doc = ElasticDocument(sess, i, text_path="article")
    p = load_pickles(sess, "../experiments/arxiv-index/pickles/default", doc.id)

    t1 = doc.text.replace("\n", " ")
    t2 = " ".join([t.text for t in p.chains])

    if t1 == t2:
        print(f"{i:04}: OF COURSE, THEY ARE THE SAME")
    else:
        print(f"{i:04}: A REMARKABLE MUTATION")

print("---------------------------------------------")

sess = Session("pubmed-index", cache_dir="../cache", use="cache")

for i in [1923, 4355, 4166, 3611, 6389, 272, 2635, 2581, 372, 6415]:
    doc = ElasticDocument(sess, i, text_path="article")
    p = load_pickles(sess, "../experiments/pubmed-index/pickles/default", doc.id)

    t1 = doc.text.replace("\n", " ")
    t2 = " ".join([t.text for t in p.chains])

    if t1 == t2:
        print(f"{i:04}: OF COURSE, THEY ARE THE SAME")
    else:
        print(f"{i:04}: A REMARKABLE MUTATION")
