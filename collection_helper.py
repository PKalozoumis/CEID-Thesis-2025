from lxml import etree
import re
import os
from collections import namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import spmatrix

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
#===================================================================================

def parse_xml(file) -> list[str]:
    f = open(file)
    contents = f.read()

    #Remove empty tags
    res = re.sub(r'<\s*/?\s*?>', '', contents)

    #Split file into documents based on the <RECORD> element
    res = re.split(r'(?=<RECORD>)', res)

    docs = []

    for doc in res[1:]:
        docs.append(etree.fromstring(doc, etree.XMLParser(recover=True)))
    
    f.close()

    return docs

#===================================================================================

def get_abstract(doc):
    abstract = (doc.xpath("//ABSTRACT/text()") + doc.xpath("//EXTRACT/text()"))

    if not abstract:
        return ""
    else:
        return abstract[0]
    
#===================================================================================

def to_json(doc):

    #print(f"Mapping {doc.xpath('//RECORDNUM/text()')[0].strip()}")
    abstract = get_abstract(doc)
    id = doc.xpath("//RECORDNUM/text()")[0].strip()

    yield {"index": {"_id": id}}

    yield {
        "paper_number": doc.xpath("//PAPERNUM/text()")[0],
        "record_number": id,
        "medline_num": doc.xpath("//MEDLINENUM/text()")[0],
        "authors": doc.xpath("//AUTHORS/AUTHOR/text()"),
        "title": doc.xpath("//TITLE/text()")[0].replace("\n", " "),
        "source": doc.xpath("//SOURCE/text()")[0],
        "major_subjects": list(map(lambda x: x.replace("-", " "), doc.xpath("//MAJORSUBJ/TOPIC/text()"))),
        "minor_subjects": list(map(lambda x: x.replace("-", " "), doc.xpath("//MINORSUBJ/TOPIC/text()"))),
        "abstract": abstract.replace("\n", " "),
        "citations": [{
                "author": cite.attrib["author"],
                "publication": cite.attrib["publication"],
                "d1": cite.attrib["d1"],
                "d2": cite.attrib["d2"],
                "d3": cite.attrib["d3"]
            }
        
            for cite in doc.xpath("//CITATIONS/CITE")
                
        ],
        "references": [{
                "author": cite.attrib["author"],
                "publication": cite.attrib["publication"],
                "d1": cite.attrib["d1"],
                "d2": cite.attrib["d2"],
                "d3": cite.attrib["d3"]
            }

            for cite in doc.xpath("//REFERENCES/CITE")
        ]
    }

#===============================================================================================

def parse_queries(collection_path: str = "collection") -> list[Query]:

    f = open(os.path.join(collection_path, "cfquery.xml"))
    res = re.split(r'(?=<QUERY>)', f.read())
    queries = map(lambda x: etree.fromstring(x, etree.XMLParser(recover=True)), res[1:])

    final_queries = []

    for query in queries:
        items = query.xpath("//Records/Item")

        docs = map(lambda item: int(item.text), items)

        scores = map(lambda item: Score(
            int(item.attrib["score"][0]),
            int(item.attrib["score"][1]),
            int(item.attrib["score"][2]),
            int(item.attrib["score"][3])
        ), items)

        final_queries.append(Query(
            int(query.xpath("//QueryNumber/text()")[0]),
            re.sub(r"\s{2,}", " ", query.xpath("//QueryText/text()")[0].strip()),
            int(query.xpath("//Results/text()")[0]),
            list(docs),
            list(scores)
        ))

    f.close()
    return final_queries