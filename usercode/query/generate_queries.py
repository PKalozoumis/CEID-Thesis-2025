import sys
import os
sys.path.append(os.path.abspath(".."))

import json

from rich.console import Console

from mypackage.elastic import Session, Query, ElasticDocument
from mypackage.helper.llm import generate_query, multi_summary_query

#======================================================================================================

def update_score(scores: dict, res: list[ElasticDocument]):
    for i, d in enumerate(res):
        if d not in scores:
            scores[d] = 10-i
        else:
            scores[d] += 10-i

#======================================================================================================

def method_01():
    console = Console()

    sess = Session("pubmed-index", credentials_path="../credentials.json", cert_path="../http_ca.crt")

    seen_docs = []
    final_result = []

    try:
        for docnum in range(159, 160):
            if docnum in seen_docs:
                continue

            doc = ElasticDocument(sess, docnum, text_path="summary")
            console.print(f"For document {doc.id:04}\n{'='*70}")

            #For each retrieved document, we will give it a score based on its rank
            #We will calculate the scores for 5 executions, each with 2 queries
            scores = {}

            for i in range(10):
                res = generate_query(doc.text)
                res = {'doc': doc.id, **res}
                console.print("The following queries were generated:")
                console.print(res)

                console.print(f"\nRound {i:02}: Perfoming retrieval with the first query:")
                s1 = Query(0, res['q1'], match_field="summary", source=["summary"]).execute(sess)
                console.print(s1)
                update_score(scores, s1)

                console.print(f"\nRound {i:02}: Perfoming retrieval with the second query:")
                s2 = Query(0, res['q2'], match_field="summary", source=["summary"]).execute(sess)
                console.print(s2)
                update_score(scores, s2)

            #Sort scores
            scores = dict(sorted([tup for tup in scores.items() if tup[0].id != doc.id], key=lambda tup: tup[1], reverse=True))
            console.print(scores)
            docs = list(scores.keys())

            #Create a query that will retrieve the documents
            res = multi_summary_query([doc.text, docs[0].text, docs[1].text])
            console.print(f"Final query: {res}")

            #Evaluating the returned queries:
            def evaluate_query(q: str, docs=list[ElasticDocument]):
                s = Query(0, q, match_field="summary", source=["."]).execute(sess)

                score = {'total': 0}

                for doc in docs:
                    try:
                        temp = 10 - s.index(doc)
                        score[doc.id] = temp
                        score['total'] += temp
                    except ValueError:
                        score[doc.id] = 0

                return score

            sc1 = evaluate_query(res['q1'], [doc, docs[0], docs[1]])
            console.print(sc1)
            sc2 = evaluate_query(res['q2'], [doc, docs[0], docs[1]])
            console.print(sc2)

            if sc1['total'] > sc2['total']:
                best = res['q1']
            else:
                best = res['q2']
            
            final_result.append({
                'query': best,
                'docs': [doc.id, docs[0].id, docs[1].id]
            })
    except KeyboardInterrupt:
        save = int(input("Do you want to save? (0/1)"))
    else:
        save = 1
    finally:
        if save == 1:
            with open("fat.json", "w") as f:
                json.dump(final_result, f)

#======================================================================================================

method_01()