from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import spmatrix
import json
from collections import namedtuple

Score = namedtuple("Score", ["s1", "s2", "s3", "s4"])
Query = namedtuple("Query", ["id", "text", "num_results", "docs", "scores"])

#===================================================================================

def tf_idf_vectorizer(docs: list[str]) -> spmatrix:
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(docs)

#===================================================================================

def count_vectorizer(docs: list[str]) -> spmatrix:
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(docs)

#===============================================================================================

def generate_examples(path, byte_offsets=None):
    """
    Yields examples.

    If byte_offsets = None, it returns all lines

    Else you can specify the subset of lines you want returned
    """

    def next_line(f):
        if byte_offsets is not None:
            for offset in byte_offsets:
                f.seek(offset)
                yield f.readline()
        else:
            yield from f

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(next_line(f)):
            if line == "":
                return
            
            d = json.loads(line)
            summary = "\n".join(d["abstract_text"])
            summary = summary.replace("<S>", "").replace("</S>", "")
            yield {
                "id": d["article_id"],
                "article": "\n".join(d["article_text"]),
                "summary": summary,
                "section_names": "\n".join(d["section_names"])
            }

#===============================================================================================

def to_bulk_format(docs):
    for doc in docs:
        yield {"index": {"_id": doc['id']}}
        yield doc