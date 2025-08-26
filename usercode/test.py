'''
Dataset generator
'''

import sys
sys.path.append("..")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--sample-size", action="store", type=int, default=500, help="The number of documents that will be evaluated")
    parser.add_argument("-nprocs", action="store", type=int, default=1, help="Number of processes")
    parser.add_argument("-s", "--start", action="store", type=int, default=0, help="List index to start the evaluation from")
    args = parser.parse_args()

from mypackage.elastic import Session
from mypackage.cluster_selection.metrics import single_document_cross_score
from mypackage.cluster_selection import RelevanceEvaluator
from mypackage.query import Query
from mypackage.storage import MongoSession

from sentence_transformers import CrossEncoder
from rich.console import Console
from rich.rule import Rule

import pandas as pd
from multiprocessing import Pool

import time

console = Console()

#=====================================================================================

def initializer():
    global cross_encoder, conn, evaluator
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2')
    conn = MongoSession(db_name="experiments_pubmed", collection="test")
    evaluator = RelevanceEvaluator(query, cross_encoder)

#=====================================================================================

def work(args):
    global cross_encoder, conn, evaluator

    proc = conn.load(sess, args[1])
    score = single_document_cross_score(args[1], evaluator)
    return {
        'idx': args[0],
        'query': query.id,
        'doc': args[1].id,
        'score': score
    }

#=====================================================================================

if __name__ == "__main__":

    if args.start >= args.sample_size:
        raise Exception("Start position out of bounds")
    console.print(f"Sample size: {args.sample_size}\nStart from: {args.start}\n")

    sess = Session("pubmed", base_path="../auth")
    queries = [
        Query(0, "What are the primary behaviours and lifestyle factors that contribute to childhood obesity", source=["summary", "article"], text_path="article")
    ]

    for query in queries:
        t = time.time()

        console.print(f"Query {query.id}: \"{query.text}\"")
        console.print(Rule())
        results = [(i, d.to_simple_document()) for i, d in enumerate(query.execute(sess, size=args.sample_size)[args.start:])]

        records = []

        with Pool(processes=args.nprocs, initializer=initializer, initargs=()) as pool:
            for i, res in enumerate(pool.imap_unordered(work, results)):
                records.append(res)
                console.print(f"{i+args.start}. [green]Document {res['doc']} score:[/green] {res['score']}")

        df = pd.DataFrame(records).sort_values(by='idx').set_index('idx')
        df.to_csv(f"query_{query.id}_results.csv", index=False)
        console.print()
        console.print(df)
        console.print(f"\nTime: {round(time.time() - t)}s\n")
