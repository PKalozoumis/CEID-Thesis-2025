from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import spmatrix
import json

#===================================================================================

def tf_idf_vectorizer(docs: list[str]) -> spmatrix:
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(docs)

#===================================================================================

def count_vectorizer(docs: list[str]) -> spmatrix:
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(docs)

#===============================================================================================

def generate_examples(path,*, doc_limit :int|None = None, byte_offsets=None):
    """
    Yields examples.

    If byte_offsets = None, it returns all lines

    Else you can specify the subset of lines you want returned
    """

    assert(not(doc_limit != None and byte_offsets != None))

    def next_line(f):
        if byte_offsets is not None:
            for offset in byte_offsets:
                f.seek(offset)
                yield f.readline()
        else:
            for i, line in enumerate(f):
                yield line

                if doc_limit and (i == doc_limit - 1):
                    break

    with open(path, encoding="utf-8") as f:
        for line in next_line(f):
            if line == "":
                return
            
            d = json.loads(line)
            summary = "\n".join(d["abstract_text"])
            summary = summary.replace("<S>", "").replace("</S>", "")
            yield {
                "article_id": d["article_id"],
                "article": "\n".join(d["article_text"]),
                "summary": summary,
                "section_names": "\n".join(d["section_names"])
            }

#===============================================================================================

def to_bulk_format(docs):
    for i, doc in enumerate(docs):
        yield {"index": {"_id": i}}
        yield doc