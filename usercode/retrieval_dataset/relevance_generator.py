import sys
sys.path.append("../..")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", action="store", type=str, default="pubmed", help="Index name")
    parser.add_argument("-n", "--sample-size", action="store", type=int, default=500, help="The number of documents that will be evaluated")
    parser.add_argument("-nprocs", action="store", type=int, default=1, help="Number of processes")
    parser.add_argument("-s", "--start", action="store", type=int, default=0, help="List index to start the evaluation from")
    parser.add_argument("-q", "--query", action="store", type=str, default=None, help="Numeric query ID or query string")
    args = parser.parse_args()

from mypackage.elastic import Session
from mypackage.cluster_selection.metrics import single_document_cross_score
from mypackage.cluster_selection import RelevanceEvaluator
from mypackage.storage import MongoSession
from mypackage.experiments import ExperimentManager

from sentence_transformers import CrossEncoder
from rich.console import Console
from rich.rule import Rule

import pandas as pd
from multiprocessing import Pool

import time

console = Console()

#=====================================================================================

def initializer():
    global conn, evaluator
    conn = MongoSession(db_name=f"experiments_{args.index}", collection="default")
    evaluator = RelevanceEvaluator(query, 'cross-encoder/ms-marco-MiniLM-L12-v2')

#=====================================================================================

def work(args):
    global conn, evaluator

    try:
        conn.load(sess, args[1])
        score, _ = single_document_cross_score(args[1], evaluator)
        return {
            'idx': args[0],
            'query': query.id,
            'doc': args[1].id,
            'score': score
        }
    except BaseException as e:
        console.print(f"[red]ERROR for document {args[1].id}[/red]: {e}")
        return {
            'idx': args[0],
            'query': query.id,
            'doc': args[1].id,
            'score': -1
        }

#=====================================================================================

if __name__ == "__main__":

    if args.start >= args.sample_size:
        raise Exception("Start position out of bounds")
    console.print(f"Sample size: {args.sample_size}\nStart from: {args.start}\n")

    sess = Session(args.index, base_path="../common")
    exp = ExperimentManager("../common/experiments.json")

    #For every query, retrieve the docs and evaluate them in parallel
    for query in exp.get_queries(args.query, sess.index_name):
        t = time.time()

        #Retrieval
        console.print(f"Query {query.id}: \"{query.text}\"")
        console.print(Rule())
        results = [(i, d.to_simple_document()) for i, d in enumerate(query.execute(sess, size=args.sample_size)[args.start:])]

        records = []

        #Parallel evaluation
        with Pool(processes=args.nprocs, initializer=initializer, initargs=()) as pool:
            for i, res in enumerate(pool.imap_unordered(work, results)):
                records.append(res)
                console.print(f"{i+args.start}. [green]Document {res['doc']} score:[/green] {res['score']}")

        #Store relevance scores into csv
        df = pd.DataFrame(records).sort_values(by='idx').set_index('idx')
        df.to_csv(f"query_{query.id}_results.csv", index=False)
        console.print()
        console.print(df)
        console.print(f"\nTime: {round(time.time() - t)}s\n")
