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

def generate_examples(path,*, doc_limit: int|None = None, byte_offsets=None, remove_duplicates: bool = False):
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

        #For every document.....
        #-----------------------------------------------------------
        for line in next_line(f):
            if line == "":
                return
            
            d = json.loads(line)
            summary = "<temp>".join(d["abstract_text"])
            summary = summary.replace("\n", "").replace("<temp>", "\n").replace("<S>", "").replace("</S>", "")

            #Remove duplicate sentences from article text
            #-----------------------------------------------------------
            seen_sentences = set()
            deduplicated = []

            if remove_duplicates:
                for sentence in d["article_text"]:
                    if len(sentence.split()) > 7:
                        if sentence in seen_sentences:
                            #print(f"{d['article_id']} IMPOSTOR DETECTED 🗣")
                            pass
                        else:
                            deduplicated.append(sentence)
                            seen_sentences.add(sentence)
            else:
                deduplicated = [s.replace("\n", "") for s in d["article_text"]]

            deduplicated = [s.replace("\n", "") for s in deduplicated]

            #Return document
            #-----------------------------------------------------------

            yield {
                "article_id": d["article_id"],
                "article": "\n".join(deduplicated),
                "summary": summary,
                "section_names": "\n".join(d["section_names"])
            }

#===============================================================================================

def to_bulk_format(docs):
    for i, doc in enumerate(docs):
        if 'usepackage' in doc['article']:
            #print(f"Document {i} has latex")
            continue

        yield {"index": {"_id": i}}
        yield doc